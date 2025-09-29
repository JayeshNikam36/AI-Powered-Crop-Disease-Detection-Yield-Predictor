import sys
import traceback
from src.utils.logger import get_logger

logger = get_logger(__name__)

class CustomException(Exception):
    def __init__(self, error_message: str, error_detail: Exception):
        super().__init__(error_message)
        self.error_message = self.get_detailed_error_message(
            error_message, error_detail
        )
        logger.error(self.error_message)  # auto log every exception

    @staticmethod
    def get_detailed_error_message(error_message: str, error_detail: Exception) -> str:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        if exc_tb:
            file_name = exc_tb.tb_frame.f_code.co_filename
            line_number = exc_tb.tb_lineno
        else:
            file_name = "Unknown"
            line_number = "Unknown"

        detailed_message = (
            f"Error occurred in script: {file_name} "
            f"at line number: {line_number} "
            f"with original exception: {repr(error_detail)} "
            f"\nContext: {error_message}"
        )
        return detailed_message

    def __str__(self):
        return self.error_message
