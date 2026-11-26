from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

def make_rf():
    return RandomForestClassifier(
        n_estimators=100, class_weight="balanced", n_jobs=-1, random_state=42
    )


def make_knn(n_neighbors=15):
    #KNN needs scaled features 
    return make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance",n_jobs=-1)
    )

def make_knn_pca(n_neighbors, n_components=8):
    StandardScaler(),
    PCA(n_components=n_components)
    KNeighborsClassifier(n_neighbors=n_neighbors,weights="distance", n_jobs=-1)