from sentence_transformers import SentenceTransformer,util
import torch


model_path=''
model = SentenceTransformer(model_path)

def construct_node_embedding(nodes):
    node_embedding = model.encode(nodes)
    return node_embedding

def classification_relationship(relationships):
    """
    :param relationships:entry relationships
    :param relationships:
    :return:
    """

    relationships_embedding = model.encode(relationships)
    similarity_matrix = util.pytorch_cos_sim(relationships_embedding, relationships_embedding)
    similarity_matrix = similarity_matrix.numpy()   # value range [-1,1]


    threshold = 0.5


    similarity_relationship = []
    dissimilarity_relationship = []
    for i in range(len(relationships)):
        for j in range(i+1, len(relationships)):
            if similarity_matrix[i][j] > threshold:
                similarity_relationship.append((relationships[i], relationships[j]))
            else:
                dissimilarity_relationship.append((relationships[i],relationships[j]))


    relationships_classification_result = {}
    for relationship in relationships:
        similarity = []
        dissimilarity = []
        for tuple in similarity_relationship:
            if relationship in tuple:
                other_element = tuple[1] if tuple[0] == relationship else tuple[0]
                similarity.append(other_element)
        for tuple in dissimilarity_relationship:
            if relationships in tuple:
                other_element = tuple[1] if tuple[0] == relationships else tuple[0]
                dissimilarity.append(other_element)
        relationships_classification_result[relationship] ={}
        relationships_classification_result[relationship]['similarity'] = similarity
        relationships_classification_result[relationship]['dissimilarity'] =dissimilarity
    print(relationships_classification_result)
    return relationships_classification_result


def construct_adj(graph,nodes_index,relations_index):
    """
    graph: sample graph
    nodes_index: entry nodes index
    relations_index: entry relations index
    """

    edge_index = []
    edge_type = []


    encode_graph = []
    for subj, rel, obj in graph:
        edge_index.append([nodes_index[subj], nodes_index[obj]])
        edge_type.append(relations_index[rel])
        encode_subj,encode_relation, encode_obj = nodes_index[subj], relations_index[rel], nodes_index[obj]
        encode_graph.append((encode_subj,encode_relation,encode_obj))


    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_type = torch.tensor(edge_type, dtype=torch.long)
    print("edge_index_len:",edge_index.size(1))
    return edge_index,edge_type,encode_graph


