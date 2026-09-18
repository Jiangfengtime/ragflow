#
#  Copyright 2024 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
import json
import logging
import re
from collections import OrderedDict, defaultdict
from dataclasses import dataclass

from rag.nlp import rag_tokenizer, query
import numpy as np
from common.doc_store.doc_store_base import MatchDenseExpr, FusionExpr, OrderByExpr, DocStoreConnection
from common.string_utils import remove_redundant_spaces
from common.float_utils import get_float
from common.constants import PAGERANK_FLD, TAG_FLD
from common.tag_feature_utils import parse_tag_features
from common import settings

from common.misc_utils import thread_pool_exec


def build_fusion_expr(topn: int, vector_similarity_weight: float = 0.3) -> FusionExpr:
    """根据向量权重构造 Infinity 的加权求和表达式。"""
    term_similarity_weight = 1 - vector_similarity_weight
    return FusionExpr(
        "weighted_sum",
        topn,
        {"weights": f"{term_similarity_weight:g},{vector_similarity_weight:g}"},
    )


def index_name(uid):
    return f"ragflow_{uid}"


class Dealer:
    def __init__(self, dataStore: DocStoreConnection):
        # dataStore 是统一文档引擎连接。使用 ES 时，文本字段走倒排索引，q_*_vec 走 HNSW
        # 向量索引，但两者命中的都是同一个以 Chunk ID 为 _id 的 ES 文档。
        self.qryr = query.FulltextQueryer()
        self.dataStore = dataStore

    @dataclass
    class SearchResult:
        total: int
        ids: list[str]
        query_vector: list[float] | None = None
        field: dict | None = None
        highlight: dict | None = None
        aggregation: list | dict | None = None
        keywords: list[str] | None = None
        group_docs: list[list] | None = None

    async def get_vector(self, txt, emb_mdl, top_k=10, num_candidates=20, similarity=0.1):
        # 将查询文本编码成与知识库 Chunk 相同向量空间的一维向量。
        # 向量维度同时决定 ES 字段名，例如 1024 维模型对应 q_1024_vec；因此跨知识库
        # 检索必须使用兼容的 Embedding 模型，否则无法查询同一个向量字段。
        qv, _ = await thread_pool_exec(emb_mdl.encode_queries, txt)
        shape = np.array(qv).shape
        if len(shape) > 1:
            raise Exception(f"Dealer.get_vector returned array's shape {shape} doesn't match expectation(exact one dimension).")
        embedding_data = [get_float(v) for v in qv]
        vector_column_name = f"q_{len(embedding_data)}_vec"
        return MatchDenseExpr(vector_column_name, embedding_data, "float", "cosine", top_k, {"similarity": similarity, "num_candidates": num_candidates})

    async def _existing_doc_ids(self, doc_ids: list[str]) -> set[str]:
        if not doc_ids:
            return set()

        unique_doc_ids = list(dict.fromkeys(doc_ids))

        def _load():
            from api.db.services.document_service import DocumentService

            return {row["id"] for row in DocumentService.get_by_ids(unique_doc_ids).dicts()}

        return await thread_pool_exec(_load)

    async def _prune_deleted_chunks(self, sres: SearchResult) -> SearchResult:
        # 临时兜底保护：部分删除链路可能只删除 MySQL 文档记录，而未完整清理检索存储中的
        # Chunk。这里过滤这些孤立 Chunk，避免聊天或检索返回已删除文档的内容；它不是主要
        # 删除机制，正常流程仍应在删除文档时同步清理检索存储。
        chunk_doc_ids = [chunk.get("doc_id") for chunk in sres.field.values() if chunk and chunk.get("doc_id")]
        if not chunk_doc_ids:
            return sres

        # 从ES候选Chunk中获取doc_id,根据这些 doc_id 查询 MySQL 的文档表。
        existing_doc_ids = await self._existing_doc_ids(chunk_doc_ids)
        if len(existing_doc_ids) == len(set(chunk_doc_ids)):
            return sres

        filtered_ids = []
        filtered_field = {}
        filtered_highlight = {} if sres.highlight else sres.highlight
        removed = 0

        for chunk_id in sres.ids:
            chunk = sres.field.get(chunk_id)
            if not chunk or chunk.get("doc_id") not in existing_doc_ids:
                removed += 1
                continue

            filtered_ids.append(chunk_id)
            filtered_field[chunk_id] = chunk
            if sres.highlight and chunk_id in sres.highlight:
                filtered_highlight[chunk_id] = sres.highlight[chunk_id]

        if removed:
            logging.warning("已修剪 %s 文档不再存在的陈旧块。", removed)

        return self.SearchResult(
            total=len(filtered_ids),
            ids=filtered_ids,
            query_vector=sres.query_vector,
            field=filtered_field,
            highlight=filtered_highlight,
            aggregation=sres.aggregation,
            keywords=sres.keywords,
            group_docs=sres.group_docs,
        )

    def get_filters(self, req):
        condition = dict()
        for key, field in {"kb_ids": "kb_id", "doc_ids": "doc_id"}.items():
            if key in req and req[key] is not None:
                condition[field] = req[key]
        # TODO(yzc)：`available_int` 可为空，但无穷大不支持可为空的列。
        for key in ["id", "knowledge_graph_kwd", "available_int", "entity_kwd", "from_entity_kwd", "to_entity_kwd", "removed_kwd"]:
            if key in req and req[key] is not None:
                condition[key] = req[key]
        if isinstance(req.get("must_not"), dict):
            condition["must_not"] = req["must_not"]
        return condition

    # `Dealer.search()` 组合以下部分：
    # 1. `kb_id`、`doc_id`、可用状态等过滤条件；
    # 2. `FulltextQueryer` 生成的全文查询；
    # 3. Embedding 模型产生的查询向量；
    # 4. `MatchDenseExpr` 发起余弦相似度检索；
    # 5. `FusionExpr` 融合全文与向量结果；
    # 6. 根据后端能力执行 Elasticsearch、Infinity 等不同实现。
    async def search(self, req, idx_names: str | list[str], kb_ids: list[str], emb_mdl=None, highlight: bool | list | None = None, rank_feature: dict | None = None, min_match: bool = True):
        if highlight is None:
            highlight = False
        # 构造过滤条件, 这些过滤条件不负责计算相似度，只负责限制搜索范围。
        filters = self.get_filters(req)
        orderBy = OrderByExpr()

        pg = int(req.get("page", 1)) - 1
        # 最终结果分页与 KNN 候选池大小彼此独立。
        ps = int(req.get("size", 30))
        offset, limit = pg * ps, ps

        knn_top_k = int(req.get("knn_top_k", 1024))
        knn_num_candidates = int(req.get("knn_num_candidates", 2048))

        # src 控制从 doc store 取回哪些字段；ES 主召回刻意不取 q_*_vec，降低网络与内存开销。
        src = req.get(
            "fields",
            [
                "docnm_kwd",
                "content_ltks",
                "kb_id",
                "img_id",
                "title_tks",
                "important_kwd",
                "position_int",
                "doc_id",
                "chunk_order_int",
                "page_num_int",
                "top_int",
                "create_timestamp_flt",
                "knowledge_graph_kwd",
                "question_kwd",
                "question_tks",
                "doc_type_kwd",
                "available_int",
                "content_with_weight",
                "mom_id",
                PAGERANK_FLD,
                TAG_FLD,
                "row_id()",
            ],
        )
        kwds = set([])

        qst = req.get("question", "")
        q_vec = []
        if not qst:
            if req.get("sort"):
                orderBy.asc("chunk_order_int")
                orderBy.asc("page_num_int")
                orderBy.asc("top_int")
                orderBy.desc("create_timestamp_flt")
            res = self.dataStore.search(src, [], filters, [], orderBy, offset, limit, idx_names, kb_ids)
            total = self.dataStore.get_total(res)
            logging.debug("Dealer.search TOTAL: {}".format(total))
        else:
            highlightFields = ["content_ltks", "title_tks"]
            if not highlight:
                highlightFields = []
            elif isinstance(highlight, list):
                highlightFields = highlight

            """
            构造全文检索条件:
            1. 对问题分词、
            2.提取关键词、
            3.构造ES query_string 查询、
            4.设置minimum_should_match
            """
            matchText, keywords = self.qryr.question(qst, min_match=(0.3 if min_match else 0))
            if emb_mdl is None:
                matchExprs = [matchText] if matchText else []
                res = await thread_pool_exec(self.dataStore.search, src, highlightFields, filters, matchExprs, orderBy, offset, limit, idx_names, kb_ids, rank_feature=rank_feature)
                total = self.dataStore.get_total(res)
                logging.debug("Dealer.search TOTAL: {}".format(total))
            else:
                # 生成问题向量并构造 KNN 查询：top_k 是期望保留的近邻数量，
                # num_candidates 是 HNSW 在每个分片中考察的候选规模，后者越大通常越准但越慢。
                matchDense = await self.get_vector(qst, emb_mdl, top_k=knn_top_k, num_candidates=knn_num_candidates, similarity=req.get("similarity", 0.1))
                q_vec = matchDense.embedding_data
                # ES 路径不再在此处获取块向量。干净的
                # 余弦分数稍后通过第二个 KNN-only 调用恢复
                # 在检索();块向量按需获取
                # 引用（参见 Dealer.fetch_chunk_vectors）。 OceanBase
                # 仍然依赖于块向量的本地重新排序，因此
                # 继续将它们拉到该后端。
                if settings.DOC_ENGINE_OCEANBASE or settings.DOC_ENGINE_SERENEDB:
                    src.append(f"q_{len(q_vec)}_vec")

                if settings.DOC_ENGINE_INFINITY:
                    vector_similarity_weight = float(req.get("vector_similarity_weight", 0.3))
                    logging.debug(
                        "Dealer.search融合：knn_top_k=%s vector_similarity_weight=%s",
                        knn_top_k,
                        vector_similarity_weight,
                    )
                    fusionExpr = build_fusion_expr(knn_top_k, vector_similarity_weight)
                elif settings.DOC_ENGINE_GAUSSDB:
                    vector_weight = req.get("vector_similarity_weight", 0.3)
                    fusionExpr = FusionExpr("weighted_sum", knn_top_k, {"weights": f"{1 - float(vector_weight)},{float(vector_weight)}"})
                else:
                    # ES 的这一组固定权重只服务于“第一阶段候选召回”，让候选集合以向量结果为主；
                    # 用户配置的 vector_similarity_weight 会在 retrieval() 的第二阶段重新打分时真正应用。
                    fusionExpr = FusionExpr("weighted_sum", knn_top_k, {"weights": "0.001,1"})
                matchExprs = [matchText, matchDense, fusionExpr] if matchText else [matchDense]

                # 对于当前 ES 实现，会生成一个同时包含 query_string（倒排索引）和
                # knn（向量索引）的请求。本次只取 rerank_candidates_count 个候选，并非最终排名。
                res = await thread_pool_exec(self.dataStore.search, src, highlightFields, filters, matchExprs, orderBy, offset, limit, idx_names, kb_ids, rank_feature=rank_feature)
                total = self.dataStore.get_total(res)
                logging.debug("Dealer.search TOTAL: {}".format(total))

                # 第一次没有候选时执行兜底召回：降低文本最低匹配比例与向量相似度门槛。
                # 指定 doc_id 时则直接读取该文档范围内的 Chunk，避免严格查询条件导致空结果。
                if total == 0:
                    if filters.get("doc_id"):
                        res = await thread_pool_exec(self.dataStore.search, src, [], filters, [], orderBy, offset, limit, idx_names, kb_ids)
                        total = self.dataStore.get_total(res)
                    else:
                        matchText, _ = self.qryr.question(qst, min_match=(0.1 if min_match else 0))
                        matchDense.extra_options["similarity"] = 0.17
                        res = await thread_pool_exec(
                            self.dataStore.search,
                            src,
                            highlightFields,
                            filters,
                            [matchText, matchDense, fusionExpr] if matchText else [matchDense],
                            orderBy,
                            offset,
                            limit,
                            idx_names,
                            kb_ids,
                            rank_feature=rank_feature,
                        )
                        total = self.dataStore.get_total(res)
                    logging.debug("Dealer.search 2 TOTAL: {}".format(total))

            for k in keywords:
                kwds.add(k)
                for kk in rag_tokenizer.fine_grained_tokenize(k).split():
                    if len(kk) < 2:
                        continue
                    if kk in kwds:
                        continue
                    kwds.add(kk)

        logging.debug(f"TOTAL: {total}")
        ids = self.dataStore.get_doc_ids(res)
        keywords = list(kwds)
        highlight = self.dataStore.get_highlight(res, keywords, "content_with_weight")
        aggs = self.dataStore.get_aggregation(res, "docnm_kwd")
        # SearchResult 保持 ids 顺序与 field[id] 一一对应；后续所有 tksim/vtsim/sim 数组
        # 都依赖这个顺序对齐，不能在重排前单独排序 field。
        return self.SearchResult(total=total, ids=ids, query_vector=q_vec, aggregation=aggs, highlight=highlight, field=self.dataStore.get_fields(res, src + ["_score"]), keywords=keywords)

    @staticmethod
    def trans2floats(txt):
        return [get_float(t) for t in txt.split("\t")]

    def insert_citations(self, answer, chunks, chunk_v, embd_mdl, tkweight=0.1, vtweight=0.9):
        assert len(chunks) == len(chunk_v)
        if not chunks:
            return answer, set([])
        pieces = re.split(r"(```)", answer)
        if len(pieces) >= 3:
            i = 0
            pieces_ = []
            while i < len(pieces):
                if pieces[i] == "```":
                    st = i
                    i += 1
                    while i < len(pieces) and pieces[i] != "```":
                        i += 1
                    if i < len(pieces):
                        i += 1
                    pieces_.append("".join(pieces[st:i]) + "\n")
                else:
                    # 句子边界正则表达式包含阿拉伯标点符号 (Ì Û Ô)
                    pieces_.extend(re.split(r"([^\|][；。？!！،؛؟۔\n]|[a-z\u0600-\u06FF][.?;!،؛؟][ \n])", pieces[i]))
                    i += 1
            pieces = pieces_
        else:
            # 句子边界正则表达式包含阿拉伯标点符号 (Ì Û Ô)
            pieces = re.split(r"([^\|][；。？!！،؛؟۔\n]|[a-z\u0600-\u06FF][.?;!،؛؟][ \n])", answer)
        for i in range(1, len(pieces)):
            if re.match(r"([^\|][；。？!！،؛؟۔\n]|[a-z\u0600-\u06FF][.?;!،؛؟][ \n])", pieces[i]):
                pieces[i - 1] += pieces[i][0]
                pieces[i] = pieces[i][1:]
        idx = []
        pieces_ = []
        for i, t in enumerate(pieces):
            if len(t) < 5:
                continue
            idx.append(i)
            pieces_.append(t)
        logging.debug("{} => {}".format(answer, pieces_))
        if not pieces_:
            return answer, set([])

        ans_v, _ = embd_mdl.encode(pieces_)
        for i in range(len(chunk_v)):
            if len(ans_v[0]) != len(chunk_v[i]):
                chunk_v[i] = [0.0] * len(ans_v[0])
                logging.warning("The dimension of query and chunk do not match: {} vs. {}".format(len(ans_v[0]), len(chunk_v[i])))

        assert len(ans_v[0]) == len(chunk_v[0]), "The dimension of query and chunk do not match: {} vs. {}".format(len(ans_v[0]), len(chunk_v[0]))

        chunks_tks = [rag_tokenizer.tokenize(self.qryr.rmWWW(ck)).split() for ck in chunks]
        cites = {}
        thr = 0.63
        while thr > 0.3 and len(cites.keys()) == 0 and pieces_ and chunks_tks:
            for i, a in enumerate(pieces_):
                sim, tksim, vtsim = self.qryr.hybrid_similarity(ans_v[i], chunk_v, rag_tokenizer.tokenize(self.qryr.rmWWW(pieces_[i])).split(), chunks_tks, tkweight, vtweight)
                mx = np.max(sim) * 0.99
                logging.debug("{} SIM: {}".format(pieces_[i], mx))
                if mx < thr:
                    continue
                cites[idx[i]] = list(set([str(ii) for ii in range(len(chunk_v)) if sim[ii] > mx]))[:4]
            thr *= 0.8

        res = ""
        seted = set([])
        for i, p in enumerate(pieces):
            res += p
            if i not in idx:
                continue
            if i not in cites:
                continue
            for c in cites[i]:
                assert int(c) < len(chunk_v)
            for c in cites[i]:
                if c in seted:
                    continue
                res += f" [ID:{c}]"
                seted.add(c)

        return res, seted

    def _tag_feature_scores(self, query_rfea, search_res):
        rank_fea = []
        if not query_rfea:
            return np.zeros(len(search_res.ids), dtype=float)

        q_denor = np.sqrt(np.sum([s * s for t, s in query_rfea.items() if t != PAGERANK_FLD]))
        if q_denor == 0:
            return np.zeros(len(search_res.ids), dtype=float)
        for i in search_res.ids:
            nor, denor = 0, 0
            if not search_res.field[i].get(TAG_FLD):
                rank_fea.append(0)
                continue
            tag_feas = parse_tag_features(search_res.field[i].get(TAG_FLD), allow_json_string=True, allow_python_literal=True)
            if not tag_feas:
                rank_fea.append(0)
                continue
            for t, sc in tag_feas.items():
                if t in query_rfea:
                    nor += query_rfea[t] * sc
                denor += sc * sc
            if denor == 0:
                rank_fea.append(0)
            else:
                rank_fea.append(nor / np.sqrt(denor) / q_denor)
        return np.array(rank_fea, dtype=float) * 10.0

    def _rank_feature_scores(self, query_rfea, search_res):
        # 对于排名特征（tag_fea）分数。
        pageranks = np.array([search_res.field[chunk_id].get(PAGERANK_FLD, 0) for chunk_id in search_res.ids], dtype=float)
        return self._tag_feature_scores(query_rfea, search_res) + pageranks

    async def _knn_scores(self, sres: "Dealer.SearchResult", idx_names: str | list[str], kb_ids: list[str]) -> dict[str, float]:
        """第二遍 ES 调用，返回
        查询嵌入和每个候选块的嵌入，过滤到
        chunk ids 原始搜索已经浮出水面。我们靠ES来做
        向量数学，因此块向量永远不会离开引擎。"""
        if not sres.ids or not sres.query_vector:
            return {}
        # 第二次查询只允许命中第一阶段已经召回的 Chunk ID。它不会扩大候选集合，
        # 作用是取得不混入全文检索 _score 的纯 KNN 分数，并避免把完整向量传回 Python。
        dim = len(sres.query_vector)
        matchDense = MatchDenseExpr(
            f"q_{dim}_vec",
            sres.query_vector,
            "float",
            "cosine",
            len(sres.ids),
            {"similarity": 0.0},
        )
        condition = {"id": list(sres.ids)}
        res = await thread_pool_exec(
            self.dataStore.search,
            [],  # 不需要 _source 字段；我们只想要 _id 和 _score
            [],
            condition,
            [matchDense],
            OrderByExpr(),
            0,
            len(sres.ids),
            idx_names,
            kb_ids,
        )
        return self.dataStore.get_scores(res)

    async def fetch_chunk_vectors(self, chunk_ids: list[str], tenant_ids: str | list[str], kb_ids: list[str], dim: int) -> dict[str, list[float]]:
        """引用时间助手：仅获取嵌入向量
        明确的块 ID 集。供需要计算的调用者使用
        本地答案与块的相似度（e.g.insert_citations）所以
        主检索路径可以继续跳过矢量传输。"""
        if not chunk_ids:
            return {}
        if isinstance(tenant_ids, str):
            idx_names = [index_name(tid) for tid in tenant_ids.split(",")]
        else:
            idx_names = [index_name(tid) for tid in tenant_ids]
        vec_field = f"q_{dim}_vec"
        res = await thread_pool_exec(
            self.dataStore.search,
            [vec_field],
            [],
            {"id": list(chunk_ids)},
            [],
            OrderByExpr(),
            0,
            len(chunk_ids),
            idx_names,
            kb_ids,
        )
        fields = self.dataStore.get_fields(res, [vec_field])
        out: dict[str, list[float]] = {}
        zero = [0.0] * dim
        for cid, doc in fields.items():
            v = doc.get(vec_field)
            if isinstance(v, str):
                v = [get_float(x) for x in v.split("\t")]
            if not isinstance(v, list) or len(v) != dim:
                v = zero
            out[cid] = v
        return out

    def rerank_with_knn(self, sres, query, knn_scores: dict[str, float], tkweight=0.3, vtweight=0.7, cfield="content_ltks", rank_feature: dict | None = None):
        """将 ES 侧 KNN 余弦相似度与本地计算项合并
        使用用户配置的权重来计算相似度。替换旧的
        ES 路径的仅本地 rerank()，这取决于运输
        块向量返回到应用程序。"""
        _, keywords = self.qryr.question(query)

        for i in sres.ids:
            if isinstance(sres.field[i].get("important_kwd", []), str):
                sres.field[i]["important_kwd"] = [sres.field[i]["important_kwd"]]
        ins_tw = []
        for i in sres.ids:
            content_ltks = list(OrderedDict.fromkeys(sres.field[i][cfield].split()))
            title_tks = [t for t in sres.field[i].get("title_tks", "").split() if t]
            question_tks = [t for t in sres.field[i].get("question_tks", "").split() if t]
            important_kwd = sres.field[i].get("important_kwd", [])
            # 第二阶段文本打分会人为重复高价值字段：正文×1、标题×2、重要词×5、
            # 预设问题×6。token_similarity 衡量加权词项覆盖率，不是 ES 原始 BM25 _score。
            tks = content_ltks + title_tks * 2 + important_kwd * 5 + question_tks * 6
            ins_tw.append(tks)

        # Python重新计算的词项匹配分数
        tksim = np.array(self.qryr.token_similarity(keywords, ins_tw), dtype=np.float64)

        # ES针对候选Chunk计算的向量相似度
        vtsim = np.array([knn_scores.get(chunk_id, 0.0) for chunk_id in sres.ids], dtype=np.float64)

        # 标签匹配分数 + PageRank分数
        rank_fea = self._rank_feature_scores(rank_feature, sres)

        # tksim = Python重新计算的词项匹配分数;
        # vtsim = ES针对候选Chunk计算的向量相似度;
        # rank_fea = 标签匹配分数 + PageRank分数
        sim = tkweight * tksim + vtweight * vtsim + rank_fea
        return sim, tksim, vtsim

    def rerank(self, sres, query, tkweight=0.3, vtweight=0.7, cfield="content_ltks", rank_feature: dict | None = None):
        _, keywords = self.qryr.question(query)
        vector_size = len(sres.query_vector)
        vector_column = f"q_{vector_size}_vec"
        zero_vector = [0.0] * vector_size
        ins_embd = []
        for chunk_id in sres.ids:
            vector = sres.field[chunk_id].get(vector_column, zero_vector)
            if isinstance(vector, str):
                vector = [get_float(v) for v in vector.split("\t")]
            ins_embd.append(vector)
        if not ins_embd:
            return [], [], []

        for i in sres.ids:
            if isinstance(sres.field[i].get("important_kwd", []), str):
                sres.field[i]["important_kwd"] = [sres.field[i]["important_kwd"]]
        ins_tw = []
        for i in sres.ids:
            content_ltks = list(OrderedDict.fromkeys(sres.field[i][cfield].split()))
            title_tks = [t for t in sres.field[i].get("title_tks", "").split() if t]
            question_tks = [t for t in sres.field[i].get("question_tks", "").split() if t]
            important_kwd = sres.field[i].get("important_kwd", [])
            tks = content_ltks + title_tks * 2 + important_kwd * 5 + question_tks * 6
            ins_tw.append(tks)

        # 对于排名特征（tag_fea）分数。
        rank_fea = self._rank_feature_scores(rank_feature, sres)

        sim, tksim, vtsim = self.qryr.hybrid_similarity(sres.query_vector, ins_embd, keywords, ins_tw, tkweight, vtweight)

        return sim + rank_fea, tksim, vtsim

    def rerank_by_model(self, rerank_mdl, sres, query, tkweight=0.3, vtweight=0.7, cfield="content_ltks", rank_feature: dict | None = None):
        # 对问题进行分词
        _, keywords = self.qryr.question(query)

        for i in sres.ids:
            if isinstance(sres.field[i].get("important_kwd", []), str):
                sres.field[i]["important_kwd"] = [sres.field[i]["important_kwd"]]
        ins_tw = []
        # 构造候选文档文本
        # 每个候选的输入由三部分构成：1. Chunk正文分词、2. 文档标题分词、3. 重要关键词
        for i in sres.ids:
            # content_ltks = 列表(OrderedDict.fromkeys(sres.field[i][cfield].split()))
            content_ltks = sres.field[i][cfield].split()
            title_tks = [t for t in sres.field[i].get("title_tks", "").split() if t]
            important_kwd = sres.field[i].get("important_kwd", [])
            tks = content_ltks + title_tks + important_kwd
            ins_tw.append(tks)

        # 转换为普通文本
        docs = [remove_redundant_spaces(" ".join(tks)) for tks in ins_tw]

        # 计算词项匹配分数: 它衡量问题关键词在候选 Chunk 中的覆盖情况。
        # 这不是 ES 返回的原始 BM25 _score，而是 RAGFlow 在 Python 中重新计算的词项匹配分数。
        tksim = self.qryr.token_similarity(keywords, ins_tw)

        # 所有供应商的 rerank_mdl.similarity() 都会返回归一化到 [0, 1] 的分数
        # （参见 RerankModel.Base.similarity），因此更换重排模型不会改变后续融合的量纲。
        # 调用 Rerank 模型
        # Rerank模型一次接收: 一个问题 + 多个候选Chunk
        # 然后分别判断问题与每个Chunk的相关程度
        # 返回值不再是Embedding 余弦相似度, 而是Rerank模型分数
        vtsim, _ = rerank_mdl.similarity(query, docs)

        # 对于排名特征（tag_fea）分数。
        # 计算标签和 PageRank 加分, 包含: 问题标签与Chunk标签匹配分数 + Chunk的PageRank分数,
        # 没有配置标签或PageRank时, 一般为0
        rank_fea = self._rank_feature_scores(rank_feature, sres)

        # 融合最终分数:
        # 最终分数 = 关键词匹配权重 × 关键词分数 + Rerank权重 × Rerank模型分数 + 标签/PageRank加分
        return tkweight * np.array(tksim) + vtweight * vtsim + rank_fea, tksim, vtsim

    def hybrid_similarity(self, ans_embd, ins_embd, ans, inst):
        return self.qryr.hybrid_similarity(ans_embd, ins_embd, rag_tokenizer.tokenize(ans).split(), rag_tokenizer.tokenize(inst).split())

    # 召回与重排
    # 1. 词法召回(全文/BM25) + 语义召回(向量/KNN)
    # 2. 融合候选
    # 3. 内置混合打分或Rerank模型
    # 4. 相似度阈值 + top N 截断
    # 5. 块 + doc_aggs
    # 当指定 rerank_mdl 时，
    async def retrieval(
        self,
        question,
        embd_mdl,
        tenant_ids,
        kb_ids,
        page,  # MUST 为 1
        page_size,  # 当指定 rerank_mdl 时为 topn
        similarity_threshold=0.2,  # 最低相似度门槛
        vector_similarity_weight=0.3,  # 语义向量相对于词法匹配的权重(相当于向量权重0.3, 文本权重0.7)
        doc_ids=None,
        aggs=True,
        rerank_mdl=None,  # 可选的专用重排模型
        highlight=False,
        rank_feature: dict | None = {PAGERANK_FLD: 10},
        trace_id=None,
        must_not: dict | None = None,
        rerank_candidates_count=64,
        knn_top_k=1024,  # 高级 knn 参数
        knn_num_candidates=2048,  # 高级 knn 参数
    ):
        """
        两阶段混合检索：search() 先从 doc store 快速召回候选；随后按后端能力取得
        独立向量分数并在应用层重排。ES 默认路径的最终文本分数是 token_similarity，
        不是直接复用第一阶段 Lucene BM25 _score。

        关键入参：``tenant_ids`` 决定 ragflow_{tenant_id} 索引，``kb_ids`` 是索引内过滤；
        ``page/page_size`` 仅用于最终结果分页，``rerank_candidates_count`` 是先取回并重排的
        候选窗口；``vector_similarity_weight`` 同时决定最终向量分数权重和全文最低匹配策略；
        ``rank_feature`` 是标签/PageRank 加分，``rerank_mdl`` 存在时替换最终的向量相似度项。

        Pagination is neither efficient nor reliable for this retrieval when rerank is enabled because the system must:
          - Retrieve more rerank candidates than the requested page_size.
          - Rerank those records to calculate similarity scores.
          - Filter out records below than the similarity threshold.
        When requesting page 2, the system must still process all candidates needed for page 1,
          resulting in a significant waste of time and computational resources. (without cache)
        Moreover, when rerank_candidates_count expands into the next retrieval window, new records are added to the candidate set and the entire set is reranked.
          That meant the previous returned pages might not be the same as the current returned pages, which is not acceptable for pagination.
        """
        # 初始化返回结果
        ranks = {"total": 0, "chunks": [], "doc_aggs": {}}
        if not question:
            return ranks

        page = max(page, 1)
        # 检查分页是否合法, 因为程序只能对前 rerank_candidates_count 个候选重新排序。
        # 如果请求范围超过候选窗口，无法保证分页正确。
        if page * page_size > rerank_candidates_count:
            raise Exception(f"rerank_candidates_count({rerank_candidates_count}) must be greater than page * page_size({page * page_size}) to ensure correct pagination.")
        # 配置了 rerank_mdl 时，只允许：page == 1
        if rerank_mdl is not None and page != 1:
            raise Exception(f"Pagination is not supported when rerank_mdl is specified. Please set page=1 to retrieve the top {page_size} results.")

        rerank_candidates_page = 1
        # 构造内部检索请求
        req = {
            "kb_ids": kb_ids,
            "doc_ids": doc_ids,
            "page": rerank_candidates_page,
            "size": rerank_candidates_count,
            "question": question,
            "vector": True,
            "similarity": similarity_threshold,
            "available_int": 1,  # available_int=1 表示只检索当前可用的 Chunk
            "vector_similarity_weight": vector_similarity_weight,
            "knn_top_k": knn_top_k,
            "knn_num_candidates": knn_num_candidates,
        }
        if isinstance(must_not, dict) and must_not:
            req["must_not"] = must_not
        logging.debug(f"[Search] page={page}, page_size={page_size}, rerank_candidates_count={rerank_candidates_count}")

        if isinstance(tenant_ids, str):
            tenant_ids = tenant_ids.split(",")

        # 确定ES索引
        idx_names = [index_name(tid) for tid in tenant_ids]
        # 根据向量检索权重，决定是否限制关键词的最低匹配比例
        min_match = vector_similarity_weight < 0.8

        # 调用search()进行第一阶段召回
        sres = await self.search(req, idx_names, kb_ids, embd_mdl, highlight, rank_feature=rank_feature, min_match=min_match)
        logging.info(
            "检索候选集已获取 跟踪ID=%s 索引=%s 知识库数=%d 候选数=%d 最小匹配=%s",
            trace_id or "-",
            idx_names,
            len(kb_ids),
            sres.total,
            min_match,
        )
        # 兜底保护: [正常情况下, 删除文档时应该同步删除ES Chunk]
        # 过滤掉所属文档已经从Mysql删除, 但仍残留在ES中的Chunk.
        sres = await self._prune_deleted_chunks(sres)
        if sres.total == 0:
            ranks["doc_aggs"] = []
            return ranks

        term_similarity_weight = 1 - vector_similarity_weight
        logging.debug(
            "[Search] 检索权重：跟踪ID=%s kb_count=%s similarity_threshold=%s vector_similarity_weight=%s full_text_weight=%s rerank_enabled=%s",
            trace_id,
            len(kb_ids),
            similarity_threshold,
            vector_similarity_weight,
            term_similarity_weight,
            bool(rerank_mdl),
        )

        # 如果配置了重排模型, 则进行重排
        if rerank_mdl and sres.total > 0:
            # 重排
            # sim: 融合后的最终分数
            # tsim: 词项匹配分数
            # vsim: Rerank模型分数
            sim, tsim, vsim = self.rerank_by_model(
                rerank_mdl,
                sres,
                question,
                term_similarity_weight,
                vector_similarity_weight,
                rank_feature=rank_feature,
            )
        else:
            if settings.DOC_ENGINE_INFINITY:
                # Infinity 会在融合前归一化每一路分数，因此这里不需要再次重排。
                sim = [sres.field[id].get("_score", 0.0) for id in sres.ids]
                sim = [s if s is not None else 0.0 for s in sim]
                tsim = sim
                vsim = sim
            elif settings.DOC_ENGINE_OCEANBASE or settings.DOC_ENGINE_SERENEDB:
                # OceanBase 仍在结果中返回 Chunk 向量，因此沿用依赖向量的本地重排逻辑。
                sim, tsim, vsim = self.rerank(
                    sres,
                    question,
                    term_similarity_weight,
                    vector_similarity_weight,
                    rank_feature=rank_feature,
                )
            elif settings.DOC_ENGINE_GAUSSDB:
                # GaussDB 在 SQL 中计算融合分数和 PageRank；标签特征在返回的候选窗口上本地叠加。
                sql_scores = [sres.field[id].get("_score", 0.0) for id in sres.ids]
                sql_scores = np.array([s if s is not None else 0.0 for s in sql_scores], dtype=np.float64)
                sim = sql_scores + self._tag_feature_scores(rank_feature, sres)
                tsim = sql_scores
                vsim = sql_scores
            else:
                # ES 路径：对第一阶段候选 ID 再执行一次纯 KNN 查询，取得干净的余弦分数，
                # 然后按用户权重与本地计算的词项相似度融合；Chunk 向量始终保留在索引中。
                # ES 默认路径的第二阶段：先取得候选的纯 KNN 分数，再与 Python 计算的
                # token_similarity 融合。这里的文本分数不是第一阶段 ES 返回的 BM25 _score。
                knn_scores = await self._knn_scores(sres, idx_names, kb_ids)
                # 真正应用用户配置的权重：
                # 最终 = (1-vector_weight)*term_score + vector_weight*knn_score + tag/PageRank。
                sim, tsim, vsim = self.rerank_with_knn(
                    sres,
                    question,
                    knn_scores,
                    term_similarity_weight,
                    vector_similarity_weight,
                    rank_feature=rank_feature,
                )

        sim_np = np.array(sim, dtype=np.float64)
        if sim_np.size == 0:
            ranks["doc_aggs"] = []
            return ranks

        # 使用稳定排序，保证分数相同时的结果顺序可复现。
        # 按 sim 降序排列
        sorted_idx = np.argsort(sim_np * -1, kind="stable")

        # similarity_threshold 在第一阶段作为 KNN 门槛使用，此处还会对重新融合后的最终分数
        # 再过滤一次。纯文本模式下分数量纲不同，因此不使用向量相似度阈值。
        post_threshold = 0.0 if vector_similarity_weight <= 0 else similarity_threshold

        # 过滤低于 similarity_threshold 的候选
        valid_idx = [int(i) for i in sorted_idx if sim_np[i] >= post_threshold]
        filtered_count = len(valid_idx)
        # total 只表示本次 rerank_candidates_count 候选窗口内通过阈值的数量，
        # 不是 ES 全索引中所有潜在命中 Chunk 的总数。
        ranks["total"] = int(filtered_count)

        if filtered_count == 0:
            ranks["doc_aggs"] = []
            return ranks

        # 截取 page_size
        begin = (page - 1) * page_size
        end = begin + page_size
        page_idx = valid_idx[begin:end]

        logging.info(
            "检索候选集排序完成 跟踪ID=%s 候选数=%d 有效数=%d 页码=%d 每页数量=%d 返回数=%d 是否重排=%s",
            trace_id or "-",
            sres.total,
            filtered_count,
            page,
            page_size,
            len(page_idx),
            bool(rerank_mdl),
        )

        dim = len(sres.query_vector)
        vector_column = f"q_{dim}_vec"
        zero_vector = [0.0] * dim

        # 构造 ranks["chunks"]
        for i in page_idx:
            id = sres.ids[i]
            chunk = sres.field[id]
            dnm = chunk.get("docnm_kwd", "")
            did = chunk.get("doc_id", "")

            position_int = chunk.get("position_int", [])
            # 主召回不再读取 Chunk 向量。Infinity 路径若已携带向量则直接使用，否则返回等维
            # 零向量占位，保持下游数据结构稳定；引用链路需要时会通过
            # Dealer.fetch_chunk_vectors() 补取真实向量。
            # 因此 ES 主链路返回的 vector 通常是等维零向量占位；需要生成引用时，
            # 调用方再通过 fetch_chunk_vectors() 按最终 Chunk ID 精确读取真实向量。
            d = {
                "chunk_id": id,
                "content_ltks": chunk["content_ltks"],
                "content_with_weight": chunk.get("content_with_weight", ""),
                "doc_id": did,
                "docnm_kwd": dnm,
                "kb_id": chunk["kb_id"],
                "important_kwd": chunk.get("important_kwd", []),
                "tag_kwd": chunk.get("tag_kwd", []),
                "image_id": chunk.get("img_id", ""),
                "similarity": float(sim_np[i]),
                "vector_similarity": float(vsim[i]),
                "term_similarity": float(tsim[i]),
                "vector": chunk.get(vector_column, zero_vector),
                "positions": position_int,
                "doc_type_kwd": chunk.get("doc_type_kwd", ""),
                "mom_id": chunk.get("mom_id", ""),
                "row_id": chunk.get("row_id()"),
            }
            if highlight and sres.highlight:
                if id in sres.highlight:
                    d["highlight"] = remove_redundant_spaces(sres.highlight[id])
                else:
                    d["highlight"] = d["content_with_weight"]
            ranks["chunks"].append(d)

        if aggs:
            # 文档聚合统计的是 valid_idx（候选窗口内通过阈值的 Chunk），用于展示各文档
            # 命中数量；它同样不是对整个 ES 结果集执行的全量聚合。
            for i in valid_idx:
                id = sres.ids[i]
                chunk = sres.field[id]
                dnm = chunk.get("docnm_kwd", "")
                did = chunk.get("doc_id", "")
                if dnm not in ranks["doc_aggs"]:
                    ranks["doc_aggs"][dnm] = {"doc_id": did, "count": 0}
                ranks["doc_aggs"][dnm]["count"] += 1

            ranks["doc_aggs"] = [
                {
                    "doc_name": k,
                    "doc_id": v["doc_id"],
                    "count": v["count"],
                }
                for k, v in sorted(
                    ranks["doc_aggs"].items(),
                    key=lambda x: x[1]["count"] * -1,
                )
            ]
        else:
            ranks["doc_aggs"] = []

        return ranks

    def sql_retrieval(self, sql, fetch_size=128, format="json"):
        tbl = self.dataStore.sql(sql, fetch_size, format)
        return tbl

    def chunk_list(
        self,
        doc_id: str,
        tenant_id: str,
        kb_ids: list[str],
        max_count=1024,
        offset=0,
        fields=["docnm_kwd", "content_with_weight", "img_id"],
        sort_by_position: bool = False,
        retrieve_all: bool = False,
    ):
        """返回文档的块。

        默认情况下，保留历史 max_count 上限。当 retrieve_all 为
        确实，保持分页直到文档存储返回的行数少于请求的行数。"""
        condition = {"doc_id": doc_id}

        fields_set = set(fields or [])
        if sort_by_position:
            for need in ("page_num_int", "position_int", "top_int"):
                if need not in fields_set:
                    fields_set.add(need)
        fields = list(fields_set)

        orderBy = OrderByExpr()
        if sort_by_position:
            orderBy.asc("page_num_int")
            orderBy.asc("position_int")
            orderBy.asc("top_int")

        res = []
        bs = 128
        p = offset
        while retrieve_all or p < max_count:
            limit = bs if retrieve_all else min(bs, max_count - p)
            if limit <= 0:
                break
            es_res = self.dataStore.search(fields, [], condition, [], orderBy, p, limit, index_name(tenant_id), kb_ids)
            dict_chunks = self.dataStore.get_fields(es_res, fields)
            for id, doc in dict_chunks.items():
                doc["id"] = id
            if dict_chunks:
                res.extend(dict_chunks.values())
            chunk_count = len(dict_chunks)
            if chunk_count == 0 or chunk_count < limit:
                break
            p += limit
        return res

    def all_tags(self, tenant_id: str, kb_ids: list[str], S=1000):
        if not self.dataStore.index_exist(index_name(tenant_id), kb_ids[0]):
            return []
        res = self.dataStore.search([], [], {}, [], OrderByExpr(), 0, 0, index_name(tenant_id), kb_ids, ["tag_kwd"])
        return self.dataStore.get_aggregation(res, "tag_kwd")

    def all_tags_in_portion(self, tenant_id: str, kb_ids: list[str], S=1000):
        res = self.dataStore.search([], [], {}, [], OrderByExpr(), 0, 0, index_name(tenant_id), kb_ids, ["tag_kwd"])
        res = self.dataStore.get_aggregation(res, "tag_kwd")
        total = np.sum([c for _, c in res])
        return {t: (c + 1) / (total + S) for t, c in res}

    def tag_content(self, tenant_id: str, kb_ids: list[str], doc, all_tags, topn_tags=3, keywords_topn=30, S=1000):
        idx_nm = index_name(tenant_id)
        match_txt = self.qryr.paragraph(doc["title_tks"] + " " + doc["content_ltks"], doc.get("important_kwd", []), keywords_topn)
        res = self.dataStore.search([], [], {}, [match_txt], OrderByExpr(), 0, 0, idx_nm, kb_ids, ["tag_kwd"])
        aggs = self.dataStore.get_aggregation(res, "tag_kwd")
        if not aggs:
            return False
        cnt = np.sum([c for _, c in aggs])
        tag_fea = sorted([(a, round(0.1 * (c + 1) / (cnt + S) / max(1e-6, all_tags.get(a, 0.0001)))) for a, c in aggs], key=lambda x: x[1] * -1)[:topn_tags]
        doc[TAG_FLD] = {a.replace(".", "_"): c for a, c in tag_fea if c > 0}
        return True

    def tag_query(self, question: str, tenant_ids: str | list[str], kb_ids: list[str], all_tags, topn_tags=3, S=1000):
        if isinstance(tenant_ids, str):
            idx_nms = index_name(tenant_ids)
        else:
            idx_nms = [index_name(tid) for tid in tenant_ids]
        match_txt, _ = self.qryr.question(question, min_match=0.0)
        # 先把问题分词并构造 ES 全文检索条件，然后只搜索标签知识库
        res = self.dataStore.search([], [], {}, [match_txt], OrderByExpr(), 0, 0, idx_nms, kb_ids, ["tag_kwd"])
        aggs = self.dataStore.get_aggregation(res, "tag_kwd")
        if not aggs:
            return {}
        cnt = np.sum([c for _, c in aggs])
        # 计算标签权重
        tag_fea = sorted([(a, round(0.1 * (c + 1) / (cnt + S) / max(1e-6, all_tags.get(a, 0.0001)))) for a, c in aggs], key=lambda x: x[1] * -1)[:topn_tags]
        return {a.replace(".", "_"): max(1, c) for a, c in tag_fea}

    async def retrieval_by_toc(self, query: str, chunks: list[dict], tenant_ids: list[str], chat_mdl, topn: int = 6):
        from rag.prompts.generator import relevant_chunks_with_toc  # 从文件顶部移出以避免循环导入

        if not chunks:
            return []
        idx_nms = [index_name(tid) for tid in tenant_ids]
        ranks, doc_id2kb_id = {}, {}
        for ck in chunks:
            if ck["doc_id"] not in ranks:
                ranks[ck["doc_id"]] = 0
            ranks[ck["doc_id"]] += ck["similarity"]
            doc_id2kb_id[ck["doc_id"]] = ck["kb_id"]
        doc_id = sorted(ranks.items(), key=lambda x: x[1] * -1.0)[0][0]
        kb_ids = [doc_id2kb_id[doc_id]]
        es_res = self.dataStore.search(["content_with_weight"], [], {"doc_id": doc_id, "toc_kwd": "toc"}, [], OrderByExpr(), 0, 128, idx_nms, kb_ids)
        toc = []
        dict_chunks = self.dataStore.get_fields(es_res, ["content_with_weight"])
        for _, doc in dict_chunks.items():
            try:
                toc.extend(json.loads(doc["content_with_weight"]))
            except Exception as e:
                logging.exception(e)
        if not toc:
            return chunks

        ids = await relevant_chunks_with_toc(query, toc, chat_mdl, topn * 2)
        if not ids:
            return chunks

        vector_size = 1024
        id2idx = {ck["chunk_id"]: i for i, ck in enumerate(chunks)}
        for cid, sim in ids:
            if cid in id2idx:
                chunks[id2idx[cid]]["similarity"] += sim
                continue
            chunk = self.dataStore.get(cid, idx_nms[0], kb_ids)
            if not chunk:
                continue
            d = {
                "chunk_id": cid,
                "content_ltks": chunk["content_ltks"],
                "content_with_weight": chunk["content_with_weight"],
                "doc_id": doc_id,
                "docnm_kwd": chunk.get("docnm_kwd", ""),
                "kb_id": chunk["kb_id"],
                "important_kwd": chunk.get("important_kwd", []),
                "image_id": chunk.get("img_id", ""),
                "similarity": sim,
                "vector_similarity": sim,
                "term_similarity": sim,
                "vector": [0.0] * vector_size,
                "positions": chunk.get("position_int", []),
                "doc_type_kwd": chunk.get("doc_type_kwd", ""),
            }
            for k in chunk.keys():
                if k[-4:] == "_vec":
                    d["vector"] = chunk[k]
                    vector_size = len(chunk[k])
                    break
            chunks.append(d)

        return sorted(chunks, key=lambda x: x["similarity"] * -1)[:topn]

    def retrieval_by_children(self, chunks: list[dict], tenant_ids: list[str]):
        if not chunks:
            return []
        idx_nms = [index_name(tid) for tid in tenant_ids]
        mom_chunks = defaultdict(list)
        i = 0
        while i < len(chunks):
            ck = chunks[i]
            mom_id = ck.get("mom_id")
            if not isinstance(mom_id, str) or not mom_id.strip():
                i += 1
                continue
            mom_chunks[ck["mom_id"]].append(chunks.pop(i))

        if not mom_chunks:
            return chunks

        if not chunks:
            chunks = []

        vector_size = 1024
        for id, cks in mom_chunks.items():
            chunk = self.dataStore.get(id, idx_nms[0], [ck["kb_id"] for ck in cks])
            if chunk is None:
                logging.warning(
                    "索引中未找到父块 '%s'；回退到 %d 子块。",
                    id,
                    len(cks),
                )
                chunks.extend(cks)
                continue
            d = {
                "chunk_id": id,
                "content_ltks": " ".join([ck["content_ltks"] for ck in cks]),
                "content_with_weight": chunk["content_with_weight"],
                "doc_id": chunk["doc_id"],
                "docnm_kwd": chunk.get("docnm_kwd", ""),
                "kb_id": chunk["kb_id"],
                "important_kwd": [kwd for ck in cks for kwd in ck.get("important_kwd", [])],
                "image_id": chunk.get("img_id", ""),
                "similarity": np.mean([ck["similarity"] for ck in cks]),
                "vector_similarity": np.mean([ck["similarity"] for ck in cks]),
                "term_similarity": np.mean([ck["similarity"] for ck in cks]),
                "vector": [0.0] * vector_size,
                "positions": chunk.get("position_int", []),
                "doc_type_kwd": chunk.get("doc_type_kwd", ""),
            }
            for k in cks[0].keys():
                if k[-4:] == "_vec":
                    d["vector"] = cks[0][k]
                    vector_size = len(cks[0][k])
                    break
            chunks.append(d)

        return sorted(chunks, key=lambda x: x["similarity"] * -1)
