# Outread-Takehome

This script parses a number of academic journals in pdf form and clusters them based on abstract similarity using a standard K-means algorithm.

Dependencies:
- numpy=1.26.2
- pandas=2.1.4
- pypdf2=2.11.1
- scikit-learn=1.3.2
- scipy=1.12.0
- matplotlib-base=3.8.2
- nltk=3.8.1
- pymupdf==1.24.7

To run the script
```bash
python cluster.py
```

Ensure you have access to pdf's of research papers and they are stored in a folder called `papers`. Or else the script will not work.

# Methodology
Development was conducted on a jupyter notebook to assist with effective visualisation and experimentation. 

### Initial API Approach
- Input data from given excel file was examined though how the data would be effectively scraped from the given doi links was unclear.
- Initial attempts were made to retrieve the data using api's and scraping available metadata were moderately successful though it did not fully meet the needs of the taks and a number of abstracts were unable to be retrieved.
- These experiments can be visualised in `test.ipynb` in the archived folder.

### After recieving pdf dataset (Pre-processing)
- Upon recieving dataset of all the pdf's abstracts were scraped using pymupdf and allocated to df row's based on the corresponding paper name extracted from pdf metadata. 
- It should be noted that extraction wasn't perfect as a few pdf's had no abstract labels or spanned multiple pages. A much more comprehensive and fine tuned approach would be required to ensure that these edge cases would effectively handled. 
- Manual input of starting words was attempted to enable detection of some of these edge cases though given a larger dataset this would be inefficient. 
- Names were normalised to ensure effective matching between the name displayed in the metadata and the one in the dataset.
- Names that didn't match were discarded.
- Presence of duplicate abstracts were found in df and were removed to have a clean df with two columns, Name and Abstract
- Text for was further cleaned to remove stop words along with stemming and lemmatizing words and any other characters, symbols or values that could cause issues. 

Inspiration of following approach drawn from:
https://www.sciencedirect.com/science/article/pii/S1877050922012947?ref=cra_js_challenge&fr=RR-1

### Vectorisation and Clustering
- The processed abstracts were vectorised using TF-IDF vectors.
- Feature names were also extracted alongside for reference in the future
- Use PCA to reduce the dimensionality of the resulting vectors, simplifying the complexity of the vectors allowing for easier classification.
- hierarchical classification was trialled initially for visualisation purposes with a dendrogram as it intuitively made sense for topics and language to have a given hierarchical structure.
- Indicated 3 very clear clusters with a silhouette Score of 0.52
- K-Means was then trialled due to the literature using it with the optimal clusters k=3 being found using the elbow method. 
- Graph seemed to indicate k could be 3,4 or 5 though testing confirmed the optimal was 3. 

### Visualisation
- Clusters visualised using PCA show 3 very clear clusters. 
