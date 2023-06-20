from disgust.dict_utils import invert

class_names = ['moral disgust', 'pathogen disgust']
class_ids_by_name = {class_name: class_id for class_id, class_name in enumerate(class_names)}


def class_name_to_class_id(class_name: str) -> int:
    """Returns the class name of a class id for the disgust case."""
    return class_ids_by_name[class_name]


def class_id_to_class_name(class_id: int) -> 'str':
    """Returns the class id of a class name for the disgust case."""
    return invert(class_ids_by_name)[class_id]
