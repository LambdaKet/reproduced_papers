"""DQuLearn reproduction: paper-specific modules.

Kept free of imports so that worker processes (``python -m lib.worker``) load
only what they use; the shared runtime is imported by the runner's caller.
"""
