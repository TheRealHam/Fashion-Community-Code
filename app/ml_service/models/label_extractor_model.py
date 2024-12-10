# -----------------------------------------------------------------
# * Prompt Label Extractor & Description Label Extractor
# ------------------------------------------------------
# * INPUT: User's prompt or Merchant's description (size: 1 ~ 50)
#   - Example: "A comfort commuting outfit with some energetic cactus design, and a number 9 design on arm."
# * OUTPUT:
#   - Style Labels: ('commuting', 'graphic', 'casual')
#   - Context Attributes: ('energetic', 'cactus', '9')
# ? USAGE:
#   - Extract and structure information from textual descriptions for consulting.
# ! POTENTIAL OPTIMIZATION: Leverage a Transformer-based encoder for improved accuracy.
# FIXME: Handle cases where descriptions are too short to produce meaningful sequence.
# -----------------------------------------------------------------
