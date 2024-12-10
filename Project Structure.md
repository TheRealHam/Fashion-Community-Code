# Models
### 1. Binary Classifier
Input: Image
Output: Clothe Score
Usage:
    - 1. Filtering dataset / Clean dataset
    - 2. Identify user upload image is standard clothe or not

### 2. Versatile Multi-function Type Classfier
Input: Image
Output: Labels(Gender, Season, Color, Master Type, Subtype)
Usage: Identify the attributes of clothing images.

### 3. Style Classifier
Input: Image
Output: Label Tempretures
Usage: Identify the style of the clothing images.

### 4. Prompt Label Extractor & Description Label Extractor
Input: User's prompt or Merchant's Description 
(A comfort commuting outfit with some energitic cactus design, and a number 9 design on arm.)
Output: StyleLabels('commuting','graphic','casual') & ContextAttributes('energitic', 'cactus', '9')
Usage: Extract information from user's prompt from consulting.
Potential Optimization: Transformers Encoder

### 5. Recommendation KNN
Matrix: Merchants' style tempreture Matrix
Input: Style attributes Tensor
Output: Top N neighboor
Usage: Recommend the similariest merchants according to style.

### 6. Recommendation TF-IDF
Input: Prompt context attributes tensor
Output: Top N most relative merchants
Usage: Recommend the similiariest merchants according to context.

