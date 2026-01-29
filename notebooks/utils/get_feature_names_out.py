from typing import List


def get_feat_names(preprocessor)->List[str]:
    try:
        feature_names = preprocessor.get_feature_names_out()
    except AttributeError:
        # Si ça échoue, on récupère les noms manuellement via les transformers
        print("Récupération manuelle des noms des features...")
        feature_names = []
        for name, transformer, columns in preprocessor.transformers_:
            if transformer == 'drop':
                continue
            if transformer == 'passthrough':
                feature_names.extend(columns)
            else:
                try:
                    feature_names.extend(transformer.get_feature_names_out(columns))
                except:
                    feature_names.extend(columns)
    return feature_names