import insightface
import roop.globals
import roop.core

FACE_ANALYSER = None


def get_face_analyser():
    global FACE_ANALYSER
    if FACE_ANALYSER is None:
        FACE_ANALYSER = insightface.app.FaceAnalysis(name='buffalo_l', providers=roop.globals.providers)
        FACE_ANALYSER.prepare(ctx_id=0, det_size=(640, 640))
    return FACE_ANALYSER


def get_face_single(img_data):
    face_analyser = get_face_analyser()
    faces = face_analyser.get(img_data)

    selected_face_index = roop.core.get_selected_face_index()

    if not faces:
        return None

    if selected_face_index is not None:
        try:
            for index in selected_face_index:
                return faces[index]
        except IndexError:
            print("Invalid face index. Using the first detected face.")
            return faces[0]
    else:
        # Default behavior: Sort faces based on bounding box x-coordinate and choose the first one
        return sorted(faces, key=lambda x: x.bbox[0])[0]


def get_face_many(img_data):
    try:
        return get_face_analyser().get(img_data)
    except IndexError:
        return None
