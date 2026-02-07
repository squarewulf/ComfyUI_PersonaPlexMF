from .personaplex_nodes import (
    PersonaPlexModelLoader, 
    PersonaPlexInference, 
    PersonaPlexConversationServer, 
    PersonaPlexStopServer,
    PersonaPlexSettings,
    PersonaPlexExternal,
    PERSONAPLEX_AVAILABLE,
    __version__,
)

NODE_CLASS_MAPPINGS = {
    "PersonaPlexModelLoader": PersonaPlexModelLoader,
    "PersonaPlexInference": PersonaPlexInference,
    "PersonaPlexConversationServer": PersonaPlexConversationServer,
    "PersonaPlexStopServer": PersonaPlexStopServer,
    "PersonaPlexSettings": PersonaPlexSettings,
    "PersonaPlexExternal": PersonaPlexExternal,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PersonaPlexModelLoader": "PersonaPlex Model Loader",
    "PersonaPlexInference": "PersonaPlex Inference",
    "PersonaPlexConversationServer": "PersonaPlex Conversation Server",
    "PersonaPlexStopServer": "PersonaPlex Stop Server",
    "PersonaPlexSettings": "PersonaPlex Settings",
    "PersonaPlexExternal": "PersonaPlex External",
}

WEB_DIRECTORY = None

if not PERSONAPLEX_AVAILABLE:
    print("[PersonaPlex] Warning: PersonaPlex dependencies not available. Run: pip install -e personaplex_src/moshi")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
