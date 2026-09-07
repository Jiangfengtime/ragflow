import { UploadFormSchemaType } from '@/components/file-upload-dialog';
import { useSetModalState } from '@/hooks/common-hooks';
import {
  useRunDocument,
  useUploadDocument,
} from '@/hooks/use-document-request';
import { getUnSupportedFilesCount } from '@/utils/document-util';
import { useCallback } from 'react';

export const useHandleUploadDocument = () => {
  const {
    visible: documentUploadVisible,
    hideModal: hideDocumentUploadModal,
    showModal: showDocumentUploadModal,
  } = useSetModalState();
  const { uploadDocument, loading } = useUploadDocument();
  const { runDocumentByIds } = useRunDocument();

  const onDocumentUploadOk = useCallback(
    async ({
      fileList,
      parseOnCreation,
      tableColumnMode,
      tableColumnRoles,
    }: UploadFormSchemaType) => {
      // 【前端上传总入口】这是“上传文件”和“启动解析”两次独立 HTTP 请求的编排层：
      // 1. uploadDocument() 先将 multipart 文件提交到后端，成功后得到 doc_id；
      // 2. 只有勾选 parseOnCreation（上传后解析）时，才用这些 doc_id 再调用 ingest；
      // 3. 未勾选时文件仍已保存，用户之后点击解析按钮会复用 useRunDocument() 启动同一链路。
      if (fileList.length > 0) {
        // Build parser_config if column roles are configured
        let parserConfig: Record<string, any> | undefined;
        if (
          tableColumnMode === 'manual' &&
          tableColumnRoles &&
          Object.keys(tableColumnRoles).length > 0
        ) {
          parserConfig = {
            table_column_mode: 'manual',
            table_column_roles: tableColumnRoles,
          };
        }
        // 第一次请求：只上传并创建 Document，不切 Chunk、不生成 Embedding。
        const ret = await uploadDocument(fileList as File[], parserConfig);

        // Check for success (code === 0) or partial success (code === 500 with some files)
        const isSuccess = ret?.code === 0;
        const isPartialSuccess = ret?.code === 500 && ret?.message;

        if (!isSuccess && !isPartialSuccess) {
          return;
        }

        // 第二次请求：POST /api/v1/documents/ingest，run=1 表示开始解析。
        // 使用服务端返回的 doc_id 关联刚上传的原文件，而不是依赖文件名。
        // 不 await 不影响后台任务：接口只负责创建 MySQL Task 并投递 Redis，真正解析由 Worker 异步完成。
        if (
          (isSuccess || isPartialSuccess) &&
          parseOnCreation && // 如果开启了文档解析
          ret.data?.length > 0
        ) {
          runDocumentByIds({
            documentIds: ret.data.map((x: any) => x.id),
            run: 1, // 开始解析
          });
        }

        if (isSuccess) {
          hideDocumentUploadModal();
          return 0;
        }

        // For partial success (code 500), check if any files were uploaded
        const count = getUnSupportedFilesCount(ret?.message);
        if (count !== fileList.length) {
          hideDocumentUploadModal();
          return 0;
        }

        return ret?.code;
      }
    },
    [uploadDocument, runDocumentByIds, hideDocumentUploadModal],
  );

  return {
    documentUploadLoading: loading,
    onDocumentUploadOk,
    documentUploadVisible,
    hideDocumentUploadModal,
    showDocumentUploadModal,
  };
};
