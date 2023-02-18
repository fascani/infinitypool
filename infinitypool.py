# -*- coding: utf-8 -*-
"""
InfinityPool Package
"""

import os
from pmaw import PushshiftAPI # https://pypi.org/project/pmaw/
import datetime as dt
import pandas as pd
from transformers import GPT2TokenizerFast
import openai
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import umap # https://umap-learn.readthedocs.io/en/latest/
import hdbscan

# Set the tokenizer
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

# Set the API key
openai.api_key = os.environ['openai-api-key']

def get_max_num_tokens():
    '''
    Max number of tokens a pre-trained NLP model can take.
    '''
    return 2046

def get_doc_model():
    '''
    String of the OpenAI pre-trained model used to calculate the embeddings.
    '''
    doc_str = 'text-embedding-ada-002'
    return doc_str

def get_embeddings(text, model):
    '''
    Calculate embeddings.

    Parameters
    ----------
    text : str
        Text to calculate the embeddings for.
    model : str
        String of the model used to calculate the embeddings.

    Returns
    -------
    list[float]
        List of the embeddings
    '''
    result = openai.Embedding.create(
      model=model,
      input=text
    )
    return result["data"][0]["embedding"]


def get_past_posts(subreddit_name,
                   before = int(dt.datetime(2023,1,15,0,0).timestamp()), 
                   after = int(dt.datetime(2023,1,1,0,0).timestamp()),
                   limit=1000, save_csv=True):
    '''
    Collect past Reddit posts from a certain subreddit. The function removes posts
    without text. The function:
        - combines title and body
        - removes past post already collected in our local csv database
        - calculates the number of tokens for title+body and remove posts that
        are too long
        - calculates the embeddings for title+body
        
    See
    https://towardsdatascience.com/how-to-collect-a-reddit-dataset-c369de539114
    https://medium.com/swlh/how-to-scrape-large-amounts-of-reddit-data-using-pushshift-1d33bde9286
    

    Parameters
    ----------
    subreddit_name : str
        Name of the subreddit.
    before : int, optional
        Date before which to collect posts. The default is int(dt.datetime(2023,1,15,0,0).timestamp()).
    after : int, optional
        Date after which to collect posts. The default is int(dt.datetime(2023,1,1,0,0).timestamp()).
    limit : int, optional
        Max number of posts to collect. The default is 1000.
    save_csv : str, optional
        Name of the local csv database to save data. The default is True.

    Returns
    -------
    posts_df : Pandas dataframe
        Database

    '''
    api = PushshiftAPI()
    posts = api.search_submissions(subreddit=subreddit_name, limit=limit, before=before, after=after)
    posts_df = pd.DataFrame(posts)
    posts_df = posts_df[['title', 'selftext']].rename(columns={'selftext': 'body'})
    num_posts_retrieved = len(posts_df)
    
    # Combine title and body
    posts_df['combined'] = "Title: " + posts_df.title.str.strip() + "; Content: " + posts_df.body.str.strip()
    
    # Remove posts without a body
    posts_df = posts_df[posts_df.body!='']
    num_posts_with_body_retrieved = len(posts_df)
    
    # Remove those that have already been collected
    orig_posts_df = read_posts_file()
    posts_df = posts_df[~(posts_df.combined.isin(orig_posts_df.combined.tolist()))]
    
    # Calculate number of tokens and remove posts that are too long
    posts_df['num_tokens'] = posts_df.combined.apply(lambda x: len(tokenizer.encode(x)))
    posts_df = posts_df[posts_df.num_tokens<get_max_num_tokens()]
    
    # Caculate the embeddings for the combined title + body
    doc_model = get_doc_model()
    posts_df['embeddings'] = posts_df.combined.apply(lambda x: get_embeddings(x, doc_model))
    
    # Caculate the embeddings for title only
    doc_model = get_doc_model()
    posts_df['title_embeddings'] = posts_df.title.apply(lambda x: get_embeddings(x, doc_model))
    
    # Add it to previous saved dataframe and dedup on title+body
    posts_df = pd.concat((orig_posts_df,posts_df),axis=0)
    posts_df.drop_duplicates(subset=['combined'], inplace=True)
    posts_df.reset_index(drop=True, inplace=True)
    
    # Print some stats
    print(f'Number of posts retrieved: {num_posts_retrieved}')
    print(f'Number of posts with a non-empty body retrieved: {num_posts_with_body_retrieved}')
    
    if save_csv:
        posts_df.to_csv('posts.csv', index=False)
            
    return posts_df
    

def vector_similarity(x, y):
    '''
    Calculate the dot product between two vectors.

    Parameters
    ----------
    x : Numpy array
    y : Numpy array

    Returns
    -------
    Float
        Dot product.

    '''
    return np.dot(np.array(x), np.array(y))

def parse_numbers(s):
  return [float(x) for x in s.strip('[]').split(',')]

def read_posts_file():
    '''
    Read the local database of subreddit posts.

    Returns
    -------
    posts_df : Pandas dataframe
    
    '''
    posts_df = pd.read_csv('posts.csv', dtype={'embeddings': object})
    for col in ['embeddings','title_embeddings']:
        posts_df[col] = posts_df[col].apply(lambda x: parse_numbers(x))
    return posts_df

def get_description_each_cluster(subreddit_name, posts_df, rev_per_cluster=3):   
    '''
    Return the description of each cluster. The approach selects a number of posts
    from each cluster and use a pre-trained NLP model to summarize these posts.
    
    See
    https://github.com/openai/openai-cookbook/blob/main/examples/Clustering.ipynb

    Parameters
    ----------
    subreddit_name : str
        Name of the subreddit.
    posts_df : Pandas dataframe
        Local database.
    rev_per_cluster : int, optional
        Number of posts to use for each cluster. The default is 3.

    Returns
    -------
    description : Pandas dataframe
        Summary of each cluster.

    '''
    
    # Reading a review which belong to each group.
    n_clusters = posts_df.cluster.max()
    print(n_clusters)
    
    description = []
    
    for i in range(n_clusters):
        print(i)
        print(f"Cluster {i} Theme:", end=" ")
    
        cluster_size = len(posts_df[posts_df.cluster == i])-1
        new_rev_per_cluster = min(rev_per_cluster, cluster_size)
    
        reviews = "\n".join(
            posts_df[posts_df.cluster == i]
            .combined.str.replace("Title: ", "")
            .str.replace("\n\nContent: ", ":  ")
            .sample(new_rev_per_cluster, random_state=42)
            .values
        )
        response = openai.Completion.create(
            engine="davinci-instruct-beta-v3",
            prompt=f'What do the following comments from subreddit/{subreddit_name} have in common?\n\nComments:\n"""\n{reviews}\n"""\n\nTheme:',
            temperature=0,
            max_tokens=64,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )
        print(response["choices"][0]["text"].replace("\n", ""))
        description.append((i, cluster_size, response["choices"][0]["text"].replace("\n", "").strip()))
        
        sample_cluster_rows = posts_df[posts_df.cluster == i].sample(new_rev_per_cluster, random_state=42)
        for j in range(new_rev_per_cluster):
            print(sample_cluster_rows.title.values[j])
    
        print("-" * 100)
        
    description = pd.DataFrame(description, columns=['cluster', '% of posts', 'description'])
    description['% of posts'] =   description['% of posts']/description['% of posts'].sum()*100
    
    return description

    
def explore_umap_hbdscan_params(posts_df, target='embeddings', umap_metric='cosine'):
    '''
    Grid search for optimal parameters for the UMAP+HDBSCAN approach to cluster text.
    
    See
    https://towardsdatascience.com/clustering-sentence-embeddings-to-identify-intents-in-short-text-48d22d3bf02e
    https://towardsdatascience.com/topic-modeling-with-bert-779f7db187e6
    

    Parameters
    ----------
    posts_df : Pandas dataframe
        Local database.
    target : Str, optional
        Column containing which embeddings we are clustering. The default is 'embeddings'.
    umap_metric : Str, optional
        Method used by UMAP. The default is 'cosine'.

    '''
    
    n_neighbors_vec = np.arange(3, 10, 1)
    n_components_vec = np.arange(2, 10, 1)
    
    print(len(n_neighbors_vec))
    
    silhouette_scores = np.zeros((len(n_neighbors_vec), len(n_components_vec)))
    calinski_harabasz_scores = np.zeros((len(n_neighbors_vec), len(n_components_vec)))
    n_clusters = np.zeros((len(n_neighbors_vec), len(n_components_vec)))
    
    Nrand = 20
    
    for i, n_neighbors in enumerate(n_neighbors_vec):
        print(f'# of neighbors: {n_neighbors}')
        for j, n_components in enumerate(n_components_vec):
            
            for nrand in np.arange(Nrand):
                umap_embeddings = (umap.UMAP(n_neighbors=n_neighbors, 
                                            n_components=n_components, 
                                            metric=umap_metric)
                                        .fit_transform(posts_df[target].tolist()))
                
                cluster = hdbscan.HDBSCAN(min_cluster_size=n_neighbors,
                                      metric='euclidean',                      
                                      cluster_selection_method='eom').fit(umap_embeddings)
                labels = cluster.labels_
                n_clusters[i, j] = n_clusters[i, j] + (len(set(labels))-1)/Nrand
                
                #posts_df['cluster'] = labels
                #posts_df['outlier'] = [True if x==-1 else True for x in labels]
    
                silhouette_scores[i, j] = silhouette_scores[i, j] + metrics.silhouette_score(umap_embeddings, labels, metric='euclidean')/Nrand
                calinski_harabasz_scores[i, j] = calinski_harabasz_scores[i, j] + metrics.calinski_harabasz_score(umap_embeddings, labels)/Nrand
            
    fig1 = plt.figure(figsize=(8, 8))
    ax1 = fig1.add_subplot(1, 1, 1)
    cs1 = ax1.contour(n_neighbors_vec, n_components_vec, silhouette_scores.T)
    ax1.set_xlabel('# of neighbors')
    ax1.set_ylabel('# of components')
    ax1.set_title('Silhouette scores')
    ax1.set_xticks(n_neighbors_vec)
    ax1.set_yticks(n_components_vec)
    ax1.clabel(cs1, cs1.levels, inline=True, fontsize=10)
    
    fig2 = plt.figure(figsize=(8, 8))
    ax2 = fig2.add_subplot(1, 1, 1)
    cs2 = ax2.contour(n_neighbors_vec, n_components_vec, calinski_harabasz_scores.T)
    ax2.set_xlabel('# of neighbors')
    ax2.set_ylabel('# of components')
    ax2.set_title('Calinksi-Harabasz scores')
    ax2.set_xticks(n_neighbors_vec)
    ax2.set_yticks(n_components_vec)
    ax2.clabel(cs2, cs2.levels, inline=True, fontsize=10)
    
    fig3 = plt.figure(figsize=(8, 8))
    ax3 = fig3.add_subplot(1, 1, 1)
    cs3 = ax3.contour(n_neighbors_vec, n_components_vec, n_clusters.T)
    ax3.set_xlabel('# of neighbors')
    ax3.set_ylabel('# of components')
    ax3.set_title('# of clusters')
    ax3.set_xticks(n_neighbors_vec)
    ax3.set_yticks(n_components_vec)
    ax3.clabel(cs3, cs3.levels, inline=True, fontsize=10)
    
    
    return fig1, ax1, cs1, fig2, ax2, cs2, fig3, ax3, cs3, silhouette_scores, calinski_harabasz_scores, n_neighbors_vec, n_components_vec

def umap_hbdscan_assign_clusters(posts_df, target='embeddings', umap_metric='cosine',
                                 n_neighbors=7, n_components=5):
    '''
    Cluster posts using the UMAP+HDBSCAN approach.

    Parameters
    ----------
    posts_df : Pandas dataframe
        Local database.
    target : str, optional
        Column containing the embeddings we want to cluster. The default is 'embeddings'.
    umap_metric : str, optional
        UMAP metric. The default is 'cosine'.
    n_neighbors : int, optional
        Number of neighbors. The default is 7.
    n_components : int, optional
        Number of components. The default is 5.

    Returns
    -------
    posts_df : Pandas dataframe
        Local database with the added column 'cluster' that indicates the cluster
        of each post.

    '''
    
    umap_embeddings = (umap.UMAP(n_neighbors=n_neighbors, 
                                n_components=n_components, 
                                metric=umap_metric)
                            .fit_transform(posts_df[target].tolist()))
    
    cluster = hdbscan.HDBSCAN(min_cluster_size=n_neighbors,
                          metric='euclidean',                      
                          cluster_selection_method='eom').fit(umap_embeddings)
    labels = cluster.labels_
    posts_df['cluster'] = labels
    posts_df['outlier'] = [True if x==-1 else True for x in labels]
    
    return posts_df

def compute_similarity_to_query(query, posts_df):
    '''
    Calculate the similarity measure for each post compared to the given query.

    Parameters
    ----------
    query : str
        Query.
    posts_df : Pandas dataframe
        Local database.

    Returns
    -------
    posts_df : Pandas dataframe
        Local database with a new column 'similarity'.

    '''
    query_model = get_doc_model()
    query_embedding = get_embeddings(query, model=query_model)
    posts_df['similarity'] = posts_df['embeddings'].apply(lambda x: vector_similarity(x, query_embedding))
    posts_df.sort_values(by='similarity', inplace=True, ascending=False)
    posts_df.reset_index(drop=True, inplace=True)
    
    return posts_df

def construct_prompt(query, subreddit_name, posts_df):
    '''
    Construct the prompt to answer the query. The prompt is composed of the
    initial query (from the user) and a context containing  the posts the most
    relevant (similar) to the query.

    Parameters
    ----------
    query : str
        Query.
    subreddit_name : str
        Name of the subreddit.
    posts_df : Pandas dataframe
        Local database.

    Returns
    -------
    prompt : str
        Prompt.

    '''
    
    MAX_SECTION_LEN = get_max_num_tokens()
    SEPARATOR = "\n* "
    separator_len = len(tokenizer.tokenize(SEPARATOR))
    
    chosen_sections = []
    chosen_sections_len = 0
    
    # Order posts_df by their similarity with the query
    posts_df = compute_similarity_to_query(query, posts_df)
     
    for section_index in range(len(posts_df)):
        # Add contexts until we run out of space.        
        document_section = posts_df.loc[section_index]
        
        chosen_sections_len += document_section.num_tokens + separator_len
        if chosen_sections_len > MAX_SECTION_LEN:
            break
            
        chosen_sections.append(SEPARATOR + document_section.body.replace("\n", " "))
        
    header = f"""
    The provided context contains online posts about {subreddit_name}.
    Please, answer the question given the context"\n\nContext:\n
    """
    prompt = header + "".join(chosen_sections) + "\n\n Q: " + query + "\n A:"
    
    return prompt

def infinity_pool(query, subreddit_name, posts_df):
    '''
    Use an OpenAI pre-trained model to answer the query about a certain subreddit
    given the local database.

    Parameters
    ----------
    query : str
        Query.
    subreddit_name : str
        Name of the subreddit.
    posts_df : Pandas dataframe
        Local database.

    Returns
    -------
    answer : str
        Answer from the model.
    prompt : str
        Actual prompt built.

    '''
    
    # Construct the prompt
    prompt = construct_prompt(query, subreddit_name, posts_df)
    
    # Ask the question with the context with GPT3 text-davinci-003
    COMPLETIONS_MODEL = "text-davinci-003"

    response = openai.Completion.create(
        prompt=prompt,
        temperature=0.5,
        max_tokens=300,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        model=COMPLETIONS_MODEL
    )

    answer = response["choices"][0]["text"].strip(" \n")
    
    return answer, prompt







