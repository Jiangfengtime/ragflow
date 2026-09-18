#
#  Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
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
import os
import json
import secrets
import logging
import re
from datetime import date

from common.constants import RAG_FLOW_SERVICE_NAME
from common.file_utils import get_project_base_directory
from common.config_utils import get_base_config, decrypt_database_config
from common.misc_utils import pip_install_torch
from common.constants import SVR_QUEUE_NAME, Storage

import rag.utils
import rag.utils.es_conn
import rag.utils.infinity_conn
import rag.utils.ob_conn
import rag.utils.opensearch_conn
import rag.utils.gaussdb_conn
from rag.utils.azure_sas_conn import RAGFlowAzureSasBlob
from rag.utils.azure_spn_conn import RAGFlowAzureSpnBlob
from rag.utils.gcs_conn import RAGFlowGCS
from rag.utils.minio_conn import RAGFlowMinio
from rag.utils.opendal_conn import OpenDALStorage
from rag.utils.redis_conn import REDIS_CONN
from rag.utils.s3_conn import RAGFlowS3
from rag.utils.oss_conn import RAGFlowOSS

from rag.nlp import search

import memory.utils.es_conn as memory_es_conn
import memory.utils.infinity_conn as memory_infinity_conn
import memory.utils.ob_conn as memory_ob_conn
import memory.utils.gaussdb_conn as memory_gaussdb_conn

TIMEZONE = os.getenv("TZ", "Asia/Shanghai")

LLM = None
LLM_FACTORY = None
LLM_BASE_URL = None
CHAT_MDL = ""
EMBEDDING_MDL = ""
RERANK_MDL = ""
ASR_MDL = ""
VISION_MDL = ""


CHAT_CFG = ""
EMBEDDING_CFG = ""
RERANK_CFG = ""
ASR_CFG = ""
VISION_CFG = ""
API_KEY = None
PARSERS = None
HOST_IP = None
HOST_PORT = None
SECRET_KEY = None
FACTORY_LLM_INFOS = None
ALLOWED_LLM_FACTORIES = None

# 元数据数据库和 DocEngine/Memory Store 都可能使用 GaussDB，而
# 针对不同的数据库、模式、兼容模式或凭据。
# 因此，元数据数据库仅读取 GAUSSDB_METADATA_*。高斯数据库
# service_conf.yaml 中的
# 部分仍然是 DOC_ENGINE=gaussdb 专有的。
GAUSSDB_ENV_DEFAULTS = {
    "name": "rag_flow",
    "user": "rag_flow",
    "password": "infini_rag_flow",
    "host": "gaussdb",
    "port": 8000,
    "schema": "public",
    "max_connections": 100,
    "stale_timeout": 30,
    "options": "-c client_encoding=UTF8 -c default_transaction_read_only=off",
}
_GAUSSDB_SCHEMA_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def normalize_database_type(database_type: str | None = None) -> str:
    # 仅标准化新的 GaussDB 值。现有数据库名称保留
    # 它们的上游拼写和查找行为。
    raw_value = database_type or "mysql"
    normalized = raw_value.strip().lower()
    if normalized in {"gaussdb", "gauss"}:
        return "gaussdb"
    return raw_value


def _get_int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _normalize_gaussdb_metadata_schema(value: str | None = None) -> str:
    # 该模式被插入到 libpq 选项中的 search_path 中。限制
    # 将其转换为普通的 SQL 标识符，因此引号、分号或额外选项不能
    # 被注入。元数据数据库当前仅接受一种模式。
    schema = (value or GAUSSDB_ENV_DEFAULTS["schema"]).strip() or GAUSSDB_ENV_DEFAULTS["schema"]
    if not _GAUSSDB_SCHEMA_PATTERN.match(schema):
        raise ValueError(f"invalid GAUSSDB_METADATA_SCHEMA: {schema}")
    return schema


def _gaussdb_metadata_options(schema: str) -> str:
    explicit_options = os.environ.get("GAUSSDB_METADATA_OPTIONS")
    if explicit_options is not None:
        # 显式值替换完整的选项字符串。高级
        # 部署可以控制search_path、编码、只读行为、
        # 和其他 libpq 选项，但还必须包含任何所需的默认值。
        return explicit_options
    return f"-c search_path={schema} {GAUSSDB_ENV_DEFAULTS['options']}"


def _gaussdb_env_config() -> dict:
    # 只读 GAUSSDB_METADATA_* 因此元数据数据库不共享
    # 连接参数为DOC_ENGINE=gaussdb。
    schema = _normalize_gaussdb_metadata_schema(os.environ.get("GAUSSDB_METADATA_SCHEMA"))
    return {
        "name": os.environ.get("GAUSSDB_METADATA_DBNAME", GAUSSDB_ENV_DEFAULTS["name"]),
        "user": os.environ.get("GAUSSDB_METADATA_USER", GAUSSDB_ENV_DEFAULTS["user"]),
        "password": os.environ.get("GAUSSDB_METADATA_PASSWORD", GAUSSDB_ENV_DEFAULTS["password"]),
        "host": os.environ.get("GAUSSDB_METADATA_HOST", GAUSSDB_ENV_DEFAULTS["host"]),
        "port": _get_int_env("GAUSSDB_METADATA_PORT", GAUSSDB_ENV_DEFAULTS["port"]),
        "max_connections": _get_int_env("GAUSSDB_METADATA_MAX_CONNECTIONS", GAUSSDB_ENV_DEFAULTS["max_connections"]),
        "stale_timeout": _get_int_env("GAUSSDB_METADATA_STALE_TIMEOUT", GAUSSDB_ENV_DEFAULTS["stale_timeout"]),
        "options": _gaussdb_metadata_options(schema),
    }


def load_database_config(database_type: str) -> dict:
    database_type = normalize_database_type(database_type)
    if database_type == "gaussdb":
        # 始终从 GAUSSDB_METADATA_* 构建 DB_TYPE=gaussdb，因此元数据
        # 连接与使用的 gaussdb 部分保持隔离
        # DOC_ENGINE=gaussdb.
        return decrypt_database_config(database=_gaussdb_env_config())
    return decrypt_database_config(name=database_type)


DATABASE_TYPE = normalize_database_type(os.getenv("DB_TYPE", "mysql"))
DATABASE = load_database_config(DATABASE_TYPE)

# 认证
AUTHENTICATION_CONF = None

# 客户端
CLIENT_AUTHENTICATION = None
HTTP_APP_KEY = None
GITHUB_OAUTH = None
FEISHU_OAUTH = None
OAUTH_CONFIG = None
DOC_ENGINE = os.getenv("DOC_ENGINE", "elasticsearch")
DOC_ENGINE_INFINITY = DOC_ENGINE.lower() == "infinity"
DOC_ENGINE_OCEANBASE = DOC_ENGINE.lower() == "oceanbase"
DOC_ENGINE_GAUSSDB = DOC_ENGINE.lower() == "gaussdb"
DOC_ENGINE_SERENEDB = DOC_ENGINE.lower() == "serenedb"


docStoreConn = None
msgStoreConn = None

retriever = None
kg_retriever = None

# 用户注册开关
REGISTER_ENABLED = 1

# SSO-only模式：隐藏密码登录表单
DISABLE_PASSWORD_LOGIN = False

# 沙箱执行器管理器
SANDBOX_HOST = None
STRONG_TEST_COUNT = int(os.environ.get("STRONG_TEST_COUNT", "8"))

SMTP_CONF = None
MAIL_SERVER = ""
MAIL_PORT = 000
MAIL_USE_SSL = True
MAIL_USE_TLS = False
MAIL_USERNAME = ""
MAIL_PASSWORD = ""
MAIL_DEFAULT_SENDER = ()
MAIL_FRONTEND_URL = ""

# 从 rag.settings 迁移
ES = {}
INFINITY = {}
AZURE = {}
S3 = {}
MINIO = {}
OB = {}
OSS = {}
OS = {}
GCS = {}
GAUSSDB = {}
SERENEDB = {}

DOC_MAXIMUM_SIZE: int = 128 * 1024 * 1024
DOC_BULK_SIZE: int = 32
EMBEDDING_BATCH_SIZE: int = 16

PARALLEL_DEVICES: int = 0

# 对象存储通过 `settings.STORAGE_IMPL` 使用。
# 业务代码只关心 `put/get/remove` 等能力，不应直接绑定 MinIO 客户端。
STORAGE_IMPL_TYPE = os.getenv("STORAGE_IMPL", "MINIO")
STORAGE_IMPL = None


def get_svr_queue_name(priority: int, suffix: str = "common") -> str:
    '''生成具有两个维度的队列名称：优先级和后缀。

    参数：
        优先级：Task优先级（0=低，1=高）
        后缀：Task 类型后缀（common/resume/graphrag/raptor/mindmap）
               目前仅使用“common”，保留其他后缀。

    返回：
        队列名称字符串

    示例：
        get_svr_queue_name(0, "common") -> "te.0.common"
        get_svr_queue_name(1, "common") -> "te.1.common"
        get_svr_queue_name(0) -> "te.0.common" # 默认后缀="common"'''
    return f"{SVR_QUEUE_NAME}.{priority}.common"


def get_svr_queue_names(suffix: str):
    """返回按优先级（从高到低）排序的队列名称。"""
    return [get_svr_queue_name(priority, suffix) for priority in [1, 0]]


def init_secret_key():
    secret_key = os.environ.get("RAGFLOW_SECRET_KEY")
    if secret_key and len(secret_key) >= 32:
        return secret_key

    # 检查是否有配置的密钥
    configured_key = get_base_config(RAG_FLOW_SERVICE_NAME, {}).get("secret_key")
    if configured_key and configured_key != str(date.today()) and len(configured_key) >= 32:
        return configured_key
    return None


def get_secret_key():
    global SECRET_KEY
    if SECRET_KEY is None:
        # 为什么需要缓存它，如果 REDIS 由于内存不足而驱逐密钥，则会生成新的密钥，导致所有请求 401
        SECRET_KEY = _get_or_create_secret_key()
    return SECRET_KEY


def _get_or_create_secret_key():
    # secret_key = os.environ.get("RAGFLOW_SECRET_KEY")
    # 如果 secret_key 且 len(secret_key) >= 32：
    # 返回 secret_key
    #
    # 检查是否有配置的秘钥
    # configured_key = get_base_config(RAG_FLOW_SERVICE_NAME, {}).get("secret_key")
    # 如果 configured_key 和 configured_key != str(date.today()) 且 len(configured_key) >= 32：
    # 返回 configured_key

    # 生成新的安全密钥并发出警告
    import logging

    generated_key = secrets.token_hex(32)
    secret_key = REDIS_CONN.get_or_create_secret_key("ragflow:system:secret_key", generated_key)
    if generated_key == secret_key:
        logging.warning("安全警告：正在使用自动生成的 SECRET_KEY。")
    return secret_key


class StorageFactory:
    storage_mapping = {
        Storage.MINIO: RAGFlowMinio,
        Storage.AZURE_SPN: RAGFlowAzureSpnBlob,
        Storage.AZURE_SAS: RAGFlowAzureSasBlob,
        Storage.AWS_S3: RAGFlowS3,
        Storage.OSS: RAGFlowOSS,
        Storage.OPENDAL: OpenDALStorage,
        Storage.GCS: RAGFlowGCS,
    }

    @classmethod
    def create(cls, storage: Storage):
        return cls.storage_mapping[storage]()


def init_settings():
    global DATABASE_TYPE, DATABASE
    DATABASE_TYPE = normalize_database_type(os.getenv("DB_TYPE", "mysql"))
    DATABASE = load_database_config(DATABASE_TYPE)

    global ALLOWED_LLM_FACTORIES, LLM_FACTORY, LLM_BASE_URL
    llm_settings = get_base_config("user_default_llm", {}) or {}
    llm_default_models = llm_settings.get("default_models", {}) or {}
    LLM_FACTORY = llm_settings.get("factory", "") or ""
    LLM_BASE_URL = llm_settings.get("base_url", "") or ""
    ALLOWED_LLM_FACTORIES = llm_settings.get("allowed_factories", None)

    global REGISTER_ENABLED
    try:
        REGISTER_ENABLED = int(os.environ.get("REGISTER_ENABLED", "1"))
    except Exception:
        pass

    global DISABLE_PASSWORD_LOGIN
    try:
        env_val = os.environ.get("DISABLE_PASSWORD_LOGIN", "").lower()
        if env_val in ("1", "true", "yes"):
            DISABLE_PASSWORD_LOGIN = True
        else:
            authentication_conf = get_base_config("authentication", {})
            DISABLE_PASSWORD_LOGIN = bool(authentication_conf.get("disable_password_login", False))
    except Exception:
        pass

    global FACTORY_LLM_INFOS
    try:
        with open(os.path.join(get_project_base_directory(), "conf", "llm_factories.json"), "r") as f:
            FACTORY_LLM_INFOS = json.load(f)["factory_llm_infos"]
    except Exception:
        FACTORY_LLM_INFOS = []

    global API_KEY
    API_KEY = llm_settings.get("api_key")

    global PARSERS
    PARSERS = llm_settings.get(
        "parsers", "naive:General,qa:Q&A,resume:Resume,manual:Manual,table:Table,paper:Paper,book:Book,laws:Laws,presentation:Presentation,picture:Picture,one:One,audio:Audio,email:Email,tag:Tag"
    )

    global CHAT_MDL, EMBEDDING_MDL, RERANK_MDL, ASR_MDL, VISION_MDL
    chat_entry = _parse_model_entry(llm_default_models.get("chat_model", CHAT_MDL))
    embedding_entry = _parse_model_entry(llm_default_models.get("embedding_model", EMBEDDING_MDL))
    rerank_entry = _parse_model_entry(llm_default_models.get("rerank_model", RERANK_MDL))
    asr_entry = _parse_model_entry(llm_default_models.get("asr_model", ASR_MDL))
    vision_entry = _parse_model_entry(llm_default_models.get("vision_model", VISION_MDL))

    global CHAT_CFG, EMBEDDING_CFG, RERANK_CFG, ASR_CFG, VISION_CFG
    CHAT_CFG = _resolve_per_model_config(chat_entry, LLM_FACTORY, API_KEY, LLM_BASE_URL)
    EMBEDDING_CFG = _resolve_per_model_config(embedding_entry, LLM_FACTORY, API_KEY, LLM_BASE_URL)
    RERANK_CFG = _resolve_per_model_config(rerank_entry, LLM_FACTORY, API_KEY, LLM_BASE_URL)
    ASR_CFG = _resolve_per_model_config(asr_entry, LLM_FACTORY, API_KEY, LLM_BASE_URL)
    VISION_CFG = _resolve_per_model_config(vision_entry, LLM_FACTORY, API_KEY, LLM_BASE_URL)

    CHAT_MDL = CHAT_CFG.get("model", "") or ""
    EMBEDDING_MDL = EMBEDDING_CFG.get("model", "") or ""
    compose_profiles = os.getenv("COMPOSE_PROFILES", "")
    if "tei-" in compose_profiles:
        EMBEDDING_MDL = os.getenv("TEI_MODEL", EMBEDDING_MDL or "BAAI/bge-small-en-v1.5")
    RERANK_MDL = RERANK_CFG.get("model", "") or ""
    ASR_MDL = ASR_CFG.get("model", "") or ""
    VISION_MDL = VISION_CFG.get("model", "") or ""

    global HOST_IP, HOST_PORT
    HOST_IP = get_base_config(RAG_FLOW_SERVICE_NAME, {}).get("host", "127.0.0.1")
    HOST_PORT = get_base_config(RAG_FLOW_SERVICE_NAME, {}).get("http_port")

    global SECRET_KEY
    SECRET_KEY = init_secret_key()

    # 认证
    authentication_conf = get_base_config("authentication", {})

    global CLIENT_AUTHENTICATION, HTTP_APP_KEY, GITHUB_OAUTH, FEISHU_OAUTH, OAUTH_CONFIG
    # 客户端
    CLIENT_AUTHENTICATION = authentication_conf.get("client", {}).get("switch", False)
    HTTP_APP_KEY = authentication_conf.get("client", {}).get("http_app_key")
    GITHUB_OAUTH = get_base_config("oauth", {}).get("github")
    FEISHU_OAUTH = get_base_config("oauth", {}).get("feishu")
    OAUTH_CONFIG = get_base_config("oauth", {})

    global DOC_ENGINE, DOC_ENGINE_INFINITY, DOC_ENGINE_OCEANBASE, DOC_ENGINE_GAUSSDB, DOC_ENGINE_SERENEDB, docStoreConn, ES, OB, OS, INFINITY, GAUSSDB, SERENEDB
    DOC_ENGINE = os.environ.get("DOC_ENGINE", "elasticsearch").strip()
    DOC_ENGINE_INFINITY = DOC_ENGINE.lower() == "infinity"
    DOC_ENGINE_OCEANBASE = DOC_ENGINE.lower() == "oceanbase"
    DOC_ENGINE_GAUSSDB = DOC_ENGINE.lower() == "gaussdb"
    DOC_ENGINE_SERENEDB = DOC_ENGINE.lower() == "serenedb"
    lower_case_doc_engine = DOC_ENGINE.lower()
    if lower_case_doc_engine == "elasticsearch":
        ES = get_base_config("es", {})
        docStoreConn = rag.utils.es_conn.ESConnection()
    elif lower_case_doc_engine == "infinity":
        INFINITY = get_base_config("infinity", {"uri": "infinity:23817", "postgres_port": 5432, "db_name": "default_db"})
        docStoreConn = rag.utils.infinity_conn.InfinityConnection()
    elif lower_case_doc_engine == "opensearch":
        OS = get_base_config("os", {})
        docStoreConn = rag.utils.opensearch_conn.OSConnection()
    elif lower_case_doc_engine == "oceanbase":
        OB = get_base_config("oceanbase", {})
        docStoreConn = rag.utils.ob_conn.OBConnection()
    elif lower_case_doc_engine == "seekdb":
        OB = get_base_config("seekdb", {})
        docStoreConn = rag.utils.ob_conn.OBConnection()
    elif lower_case_doc_engine == "gaussdb":
        GAUSSDB = get_base_config("gaussdb", {})
        docStoreConn = rag.utils.gaussdb_conn.GaussDBConnection()
    elif lower_case_doc_engine == "serenedb":
        SERENEDB = get_base_config("serenedb", {})
        # 延迟导入，因此 psycopg2/SereneDB 仅在选择时才会被触摸。
        from rag.utils import serenedb_conn

        docStoreConn = serenedb_conn.SereneDBConnection()
    else:
        raise Exception(f"Not supported doc engine: {DOC_ENGINE}")

    global msgStoreConn
    # 使用相同的引擎进行消息存储
    if DOC_ENGINE == "elasticsearch":
        ES = get_base_config("es", {})
        msgStoreConn = memory_es_conn.ESConnection()
    elif DOC_ENGINE == "infinity":
        INFINITY = get_base_config("infinity", {"uri": "infinity:23817", "postgres_port": 5432, "db_name": "default_db"})
        msgStoreConn = memory_infinity_conn.InfinityConnection()
    elif lower_case_doc_engine in ["oceanbase", "seekdb"]:
        msgStoreConn = memory_ob_conn.OBConnection()
    elif lower_case_doc_engine == "gaussdb":
        # Memory Store 使用消息表专用适配器。它读取
        # 与 GaussDB 相同的配置并共享惰性连接池
        # docStoreConn，但保留其自己的表布局和查询语义。
        msgStoreConn = memory_gaussdb_conn.GaussDBMemoryConnection()

    global AZURE, S3, MINIO, OSS, GCS
    if STORAGE_IMPL_TYPE in ["AZURE_SPN", "AZURE_SAS"]:
        AZURE = get_base_config("azure", {})
    elif STORAGE_IMPL_TYPE == "AWS_S3":
        S3 = get_base_config("s3", {})
    elif STORAGE_IMPL_TYPE == "MINIO":
        MINIO = decrypt_database_config(name="minio")
    elif STORAGE_IMPL_TYPE == "OSS":
        OSS = get_base_config("oss", {})
    elif STORAGE_IMPL_TYPE == "GCS":
        GCS = get_base_config("gcs", {})

    global STORAGE_IMPL
    storage_impl = StorageFactory.create(Storage[STORAGE_IMPL_TYPE])

    # 定义加密设置
    crypto_enabled = os.environ.get("RAGFLOW_CRYPTO_ENABLED", "false").lower() == "true"

    # 检查是否启用加密
    if crypto_enabled:
        try:
            from rag.utils.encrypted_storage import create_encrypted_storage

            algorithm = os.environ.get("RAGFLOW_CRYPTO_ALGORITHM", "aes-256-cbc")
            crypto_key = os.environ.get("RAGFLOW_CRYPTO_KEY")

            STORAGE_IMPL = create_encrypted_storage(storage_impl, algorithm=algorithm, key=crypto_key, encryption_enabled=crypto_enabled)
        except Exception as e:
            logging.error(f"Failed to initialize encrypted storage: {e}")
            STORAGE_IMPL = storage_impl
    else:
        STORAGE_IMPL = storage_impl

    global retriever, kg_retriever
    # retriever 不是另一套存储：Dealer 持有同一个 docStoreConn。当前配置为 ES 时，
    # 全文 query_string 与 dense_vector KNN 都最终进入 ESConnection.search()。
    retriever = search.Dealer(docStoreConn)
    from rag.graphrag import search as kg_search

    kg_retriever = kg_search.KGSearch(docStoreConn)

    global SANDBOX_HOST
    if int(os.environ.get("SANDBOX_ENABLED", "0")):
        SANDBOX_HOST = os.environ.get("SANDBOX_HOST", "sandbox-executor-manager")

    global SMTP_CONF
    SMTP_CONF = get_base_config("smtp", {})

    global MAIL_SERVER, MAIL_PORT, MAIL_USE_SSL, MAIL_USE_TLS, MAIL_USERNAME, MAIL_PASSWORD, MAIL_DEFAULT_SENDER, MAIL_FRONTEND_URL
    MAIL_SERVER = SMTP_CONF.get("mail_server", "")
    MAIL_PORT = SMTP_CONF.get("mail_port", 000)
    MAIL_USE_SSL = SMTP_CONF.get("mail_use_ssl", True)
    MAIL_USE_TLS = SMTP_CONF.get("mail_use_tls", False)
    MAIL_USERNAME = SMTP_CONF.get("mail_username", "")
    MAIL_PASSWORD = SMTP_CONF.get("mail_password", "")
    mail_default_sender = SMTP_CONF.get("mail_default_sender", [])
    if mail_default_sender and len(mail_default_sender) >= 2:
        MAIL_DEFAULT_SENDER = (mail_default_sender[0], mail_default_sender[1])
    MAIL_FRONTEND_URL = SMTP_CONF.get("mail_frontend_url", "")

    global DOC_MAXIMUM_SIZE, DOC_BULK_SIZE, EMBEDDING_BATCH_SIZE
    DOC_MAXIMUM_SIZE = int(os.environ.get("MAX_CONTENT_LENGTH", 128 * 1024 * 1024))
    DOC_BULK_SIZE = int(os.environ.get("DOC_BULK_SIZE", 32))
    EMBEDDING_BATCH_SIZE = int(os.environ.get("EMBEDDING_BATCH_SIZE", 16))

    os.environ["DOTNET_SYSTEM_GLOBALIZATION_INVARIANT"] = "1"


def check_and_install_torch():
    global PARALLEL_DEVICES
    try:
        pip_install_torch()
        import torch.cuda

        PARALLEL_DEVICES = torch.cuda.device_count()
        logging.info(f"found {PARALLEL_DEVICES} gpus")
    except Exception:
        logging.info("无法导入包“torch”")


def _parse_model_entry(entry):
    if isinstance(entry, str):
        return {"name": entry, "factory": None, "api_key": None, "base_url": None}
    if isinstance(entry, dict):
        name = entry.get("name") or entry.get("model") or ""
        return {
            "name": name,
            "factory": entry.get("factory"),
            "api_key": entry.get("api_key"),
            "base_url": entry.get("base_url"),
        }
    return {"name": "", "factory": None, "api_key": None, "base_url": None}


def _resolve_per_model_config(entry_dict, backup_factory, backup_api_key, backup_base_url):
    name = (entry_dict.get("name") or "").strip()
    m_factory = entry_dict.get("factory") or backup_factory or ""
    m_api_key = entry_dict.get("api_key") or backup_api_key or ""
    m_base_url = entry_dict.get("base_url") or backup_base_url or ""

    if name and "@" not in name and m_factory:
        name = f"{name}@{m_factory}"

    return {
        "model": name,
        "factory": m_factory,
        "api_key": m_api_key,
        "base_url": m_base_url,
    }


def print_rag_settings():
    logging.info(f"MAX_CONTENT_LENGTH: {DOC_MAXIMUM_SIZE}")
    logging.info(f"MAX_FILE_COUNT_PER_USER: {int(os.environ.get('MAX_FILE_NUM_PER_USER', 0))}")
