# Models

### 1. Binary Classifier
- **Input**: Image
- **Output**: Clothe Score
- **Usage**:
  - Filtering dataset / Clean dataset
  - Identify if the user-uploaded image is a standard clothe or not

### 2. Versatile Multi-function Type Classifier
- **Input**: Image
- **Output**: Labels (Gender, Season, Color, Master Type, Subtype)
- **Usage**: Identify the attributes of clothing images

### 3. Style Classifier
- **Input**: Image
- **Output**: Label Temperatures
- **Usage**: Identify the style of the clothing images

### 4. Prompt Label Extractor & Description Label Extractor
- **Input**: User's prompt or Merchant's Description 
  - Example: "A comfort commuting outfit with some energetic cactus design, and a number 9 design on arm."
- **Output**:
  - **Style Labels**: ('commuting', 'graphic', 'casual')
  - **Context Attributes**: ('energetic', 'cactus', '9')
- **Usage**: Extract information from the user's prompt for consulting
- **Potential Optimization**: Transformers Encoder

### 5. Recommendation KNN
- **Matrix**: Merchants' style temperature Matrix
- **Input**: Style attributes Tensor
- **Output**: Top N neighbors
- **Usage**: Recommend the most similar merchants according to style

### 6. Recommendation TF-IDF
- **Input**: Prompt context attributes tensor
- **Output**: Top N most relevant merchants
- **Usage**: Recommend the most similar merchants according to context
