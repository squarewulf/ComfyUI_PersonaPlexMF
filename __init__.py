from .personaplex_nodes import (
    PersonaPlexModelLoader, 
    PersonaPlexInference, 
    PersonaPlexConversationServer, 
    PersonaPlexSettings,
    PERSONAPLEX_AVAILABLE
)

NODE_CLASS_MAPPINGS = {
    "PersonaPlexModelLoader": PersonaPlexModelLoader,
    "PersonaPlexInference": PersonaPlexInference,
    "PersonaPlexConversationServer": PersonaPlexConversationServer,
    "PersonaPlexSettings": PersonaPlexSettings,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PersonaPlexModelLoader": "PersonaPlex Model Loader",
    "PersonaPlexInference": "PersonaPlex Inference",
    "PersonaPlexConversationServer": "PersonaPlex Conversation Server",
    "PersonaPlexSettings": "PersonaPlex Settings",
}

WEB_DIRECTORY = None

if not PERSONAPLEX_AVAILABLE:
    print("[PersonaPlex] Warning: PersonaPlex dependencies not available. Run: pip install -e personaplex_src/moshi")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
