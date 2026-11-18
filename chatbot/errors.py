"""Custom exceptions with user-friendly messages for the chatbot application."""


class ChatbotError(Exception):
    """Base exception for all chatbot errors."""

    default_user_message: str = "An unexpected error occurred. Please try again."

    def __init__(self, message: str = "", user_message: str | None = None) -> None:
        super().__init__(message or user_message or self.default_user_message)
        self.user_message: str = user_message or (message if message else self.default_user_message)


class AuthenticationFailedError(ChatbotError):
    """Raised when authentication with the LLM provider fails."""

    default_user_message: str = "Authentication failed. Please verify your LLM_API_KEY in the .env file."


class RateLimitedError(ChatbotError):
    """Raised when the LLM provider returns a rate limit error."""

    default_user_message: str = "Rate limit reached. Please wait a moment before sending another message."


class ConnectionFailedError(ChatbotError):
    """Raised when network connection or request to the LLM provider fails."""

    default_user_message: str = "Failed to connect to the LLM provider. Please check your internet connection or base URL."


class ModelError(ChatbotError):
    """Raised when model is not found or request format is rejected."""

    default_user_message: str = "The specified model was not found or request was invalid. Please check your LLM_MODEL setting."


class LLMError(ChatbotError):
    """General LLM provider error."""

    default_user_message: str = "An error occurred while communicating with the LLM service."


class InvalidInputError(ChatbotError):
    """Raised when user input fails validation."""

    default_user_message: str = "Invalid input provided."


class ConfigError(ChatbotError):
    """Raised when application configuration is missing or invalid."""

    default_user_message: str = "Configuration error. Missing or invalid settings."
