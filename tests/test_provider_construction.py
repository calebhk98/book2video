from book2video.providers.anthropic_llm import AnthropicLLMProvider
from book2video.providers.comfyui_video import ComfyUIVideoProvider
from book2video.providers.fal_video import FalVideoProvider
from book2video.providers.gemini_llm import GeminiLLMProvider
from book2video.providers.ollama_llm import OllamaLLMProvider
from book2video.providers.openai_compatible_llm import OpenAICompatibleLLMProvider
from book2video.providers.openai_llm import OpenAILLMProvider
from book2video.providers.replicate_video import ReplicateVideoProvider
from book2video.providers.runway_video import RunwayVideoProvider
from book2video.providers.command_video import CommandVideoProvider


def test_builtin_providers_construct_without_network_calls():
    assert OllamaLLMProvider(model="local").provider_id == "ollama"
    assert OpenAICompatibleLLMProvider(model="local").provider_id == "openai-compatible"
    assert OpenAILLMProvider(model="online").provider_id == "openai"
    assert AnthropicLLMProvider(model="online").provider_id == "anthropic"
    assert GeminiLLMProvider(model="online").provider_id == "gemini"

    assert CommandVideoProvider(command=["true"]).provider_id == "command-video"
    assert ComfyUIVideoProvider(workflow_path="workflow.json").provider_id == "comfyui"
    assert RunwayVideoProvider().provider_id == "runway"
    assert ReplicateVideoProvider(model="owner/model").provider_id == "replicate"
    assert FalVideoProvider(model="owner/model").provider_id == "fal"
