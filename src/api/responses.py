# 通用 API 响应模型：为所有接口提供一致的成功/错误响应结构，确保 OpenAPI 文档完整。
# 所有路由应通过 response_model 引用这些模型，使 /docs 可以正确展示各状态码的响应 schema。

from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    """统一的错误响应格式，用于 400/404/422/500 等异常状态码。"""

    detail: str = Field(..., description="人类可读的错误描述")
    message: str | None = Field(None, description="可选的详细错误信息（生产环境应置为 None）")


class ValidationErrorDetail(BaseModel):
    """单个字段的校验错误信息。"""

    loc: list[str | int] = Field(..., description="错误位置（字段路径）")
    msg: str = Field(..., description="校验错误消息")
    type: str = Field(..., description="错误类型标识")


class ValidationErrorResponse(BaseModel):
    """422 校验错误的响应格式。"""

    detail: list[ValidationErrorDetail] = Field(..., description="校验错误详情列表")


class HealthResponse(BaseModel):
    """GET /health 健康检查成功响应。"""

    status: str = Field(..., description="服务状态", examples=["healthy"])
    service: str = Field(..., description="服务名称", examples=["enterprise-rag"])


class DeleteResponse(BaseModel):
    """DELETE /documents/{doc_id} 删除文档成功响应。"""

    message: str = Field(..., description="操作结果描述")
    doc_id: str = Field(..., description="已删除的文档 ID")


class BulkImportItemResult(BaseModel):
    """批量导入中单个文件的处理结果。"""

    filename: str = Field(..., description="文件名", examples=["report.pdf"])
    status: str = Field(..., description="处理状态：success / failed / skipped", examples=["success"])
    doc_id: str | None = Field(None, description="成功时返回的文档 ID")
    chunk_count: int | None = Field(None, description="成功时返回的分块数量")
    error: str | None = Field(None, description="失败时的错误信息（成功时为 null）")


class BulkImportResponse(BaseModel):
    """POST /documents/bulk-import 批量导入的响应格式。"""

    total: int = Field(..., description="提交的文件总数", examples=[12])
    success_count: int = Field(..., description="成功导入的文件数")
    failed_count: int = Field(..., description="导入失败的文件数")
    skipped_count: int = Field(..., description="跳过的文件数（格式不支持等）")
    results: list[BulkImportItemResult] = Field(..., description="每个文件的处理结果详情")
