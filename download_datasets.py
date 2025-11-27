"""
Script pour télécharger et préparer les datasets améliorés.
"""

import os
import subprocess
import sys

def download_ferplus():
    """Télécharge les labels FER+ depuis GitHub."""
    print("="*60)
    print("Téléchargement des labels FER+")
    print("="*60)
    
    data_dir = './data'
    os.makedirs(data_dir, exist_ok=True)
    
    ferplus_file = os.path.join(data_dir, 'fer2013new.csv')
    
    if os.path.exists(ferplus_file):
        print(f"✓ FER+ labels déjà présents: {ferplus_file}")
        return True
    
    print("Clonage du repo FERPlus...")
    
    try:
        # Cloner le repo FERPlus
        subprocess.run([
            'git', 'clone', '--depth', '1',
            'https://github.com/microsoft/FERPlus.git',
            'temp_ferplus'
        ], check=True)
        
        # Copier le fichier de labels
        import shutil
        src = os.path.join('temp_ferplus', 'data', 'fer2013new.csv')
        if os.path.exists(src):
            shutil.copy(src, ferplus_file)
            print(f"✓ Labels copiés vers: {ferplus_file}")
        
        # Nettoyer
        shutil.rmtree('temp_ferplus', ignore_errors=True)
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"✗ Erreur lors du clonage: {e}")
        print("\nTéléchargez manuellement depuis:")
        print("https://github.com/microsoft/FERPlus/blob/master/data/fer2013new.csv")
        return False
    except Exception as e:
        print(f"✗ Erreur: {e}")
        return False


def check_fer2013():
    """Vérifie que FER2013 est présent."""
    print("\n" + "="*60)
    print("Vérification de FER2013")
    print("="*60)
    
    fer_file = './data/fer2013/fer2013.csv'
    
    if os.path.exists(fer_file):
        import pandas as pd
        df = pd.read_csv(fer_file)
        print(f"✓ FER2013 trouvé: {len(df)} images")
        
        # Afficher la distribution
        print("\nDistribution des émotions:")
        emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        counts = df['emotion'].value_counts().sort_index()
        for i, count in counts.items():
            print(f"  {emotions[i]}: {count}")
        
        return True
    else:
        print("✗ FER2013 non trouvé!")
        print("\nTéléchargez depuis Kaggle:")
        print("https://www.kaggle.com/datasets/msambare/fer2013")
        print("\nOu utilisez la commande (nécessite kaggle CLI):")
        print("kaggle datasets download -d msambare/fer2013")
        return False


def analyze_dataset_quality():
    """Analyse la qualité du dataset et affiche des recommandations."""
    print("\n" + "="*60)
    print("Analyse de qualité du dataset")
    print("="*60)
    
    fer_file = './data/fer2013/fer2013.csv'
    ferplus_file = './data/fer2013/fer2013new.csv'
    
    if not os.path.exists(fer_file):
        print("FER2013 non disponible pour l'analyse.")
        return
    
    import pandas as pd
    import numpy as np
    
    fer = pd.read_csv(fer_file)
    
    # Distribution par usage
    print("\nDistribution par split:")
    print(fer['Usage'].value_counts())
    
    # Problème de déséquilibre
    emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    counts = fer['emotion'].value_counts().sort_index()
    
    print("\n⚠️  Problèmes identifiés:")
    
    min_class = counts.idxmin()
    max_class = counts.idxmax()
    ratio = counts[max_class] / counts[min_class]
    
    print(f"  - Déséquilibre: {emotions[max_class]} a {ratio:.1f}x plus d'exemples que {emotions[min_class]}")
    print(f"  - {emotions[1]} (Disgust) n'a que {counts[1]} exemples ({100*counts[1]/len(fer):.1f}%)")
    
    # Vérifier FER+
    if os.path.exists(ferplus_file):
        print("\n✓ FER+ disponible - les annotations corrigées seront utilisées")
        
        ferplus = pd.read_csv(ferplus_file)
        
        # Compter les images où les annotateurs sont en désaccord
        vote_cols = ['neutral', 'happiness', 'surprise', 'sadness', 
                     'anger', 'disgust', 'fear', 'contempt', 'unknown', 'NF']
        
        if all(col in ferplus.columns for col in vote_cols):
            votes = ferplus[vote_cols].values
            max_votes = votes.max(axis=1)
            total_votes = votes.sum(axis=1)
            
            # Accord si > 50% des votes pour une émotion
            agreement = (max_votes / total_votes) > 0.5
            print(f"  - {agreement.sum()}/{len(agreement)} images avec accord majoritaire ({100*agreement.mean():.1f}%)")
    else:
        print("\n⚠️  FER+ non disponible - utilisez les annotations originales (moins précises)")
    
    # Recommandations
    print("\n" + "="*60)
    print("RECOMMANDATIONS")
    print("="*60)
    print("""
1. ✓ Téléchargez FER+ pour des annotations plus précises
   
2. ✓ Utilisez le WeightedRandomSampler pour équilibrer les classes
   
3. ⭐ Pour de meilleurs résultats, considérez:
   - AffectNet (450K images) - Le meilleur mais payant
   - RAF-DB (30K images) - Gratuit pour recherche
   
4. ✓ Data augmentation est cruciale pour FER2013
   
5. ✓ Utilisez l'égalisation d'histogramme pour normaliser l'éclairage
""")


if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║          PRÉPARATION DES DATASETS D'ÉMOTIONS                 ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    
    # Vérifier FER2013
    fer_ok = check_fer2013()
    
    if fer_ok:
        # Télécharger FER+
        download_ferplus()
        
        # Analyser la qualité
        analyze_dataset_quality()
    else:
        print("\n⚠️  Veuillez d'abord télécharger FER2013 depuis Kaggle.")
