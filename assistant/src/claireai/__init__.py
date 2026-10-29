"""ClaireAI assistant runtime (Track A).

Local-first: every component in this package talks to a model through the
provider-neutral ``claireai.brain.Brain`` interface, and the default adapter is
a locally installed Ollama runtime on loopback.
"""

__version__ = "0.1.0"
