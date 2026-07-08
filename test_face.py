from fastapi import FastAPI, UploadFile, File, Depends, Form, Body
from typing import List, Annotated
from schemes import Person
from app import *
import cv2
import numpy as np
import os
from collections import Counter
import shutil
import traceback
from string import ascii_letters, digits, punctuation
import uuid
import uvicorn


app = FastAPI(
    title="Face Recognition API",
    description="High-concurrency API for face recognition in images",
    version="1.0.0",
)

collection_name = "face_embeddings_v2"
facedet = RetinanetEngine(**CONFIG_FACEDET)
arcface = ArcfaceEngine(**CONFIG_GHOSTFACE)
spoofingdet = FakeFaceEngine(f"{str(ROOT)}/weights/spoof.onnx_b8_gpu0_fp16.engine")
yolovaliface = YoloFaceEngine(f"{str(ROOT)}/weights/face_part-trt.engine", imgsz=(640,640), conf_thres=0.25, device='cuda')
ALLOWED_CODE_CHARS = ascii_letters + digits + punctuation

def _ensure_collection():
    try:
        qdrantclient.get_collection(collection_name)
    except Exception:
        qdrantclient.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=512, distance=models.Distance.COSINE),
        )


@app.post("/api/spoofingCheck")
async def spoofing_check(image: UploadFile = File(...)):
    try:
        image_byte = await image.read()
        nparr = np.frombuffer(image_byte, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        results_face = facedet.inference([img])
        dets, miss_det, croped_image = results_face
        if len(croped_image) == 0:
            return {"success": False, "error_code": 8001, "error": "Don't find any face"}
        for croped in croped_image:
            t_valid = time.time()
            _, valid_scores, valid_cls_inds = yolovaliface.infer(croped[None,...])
            logger.info(f"---------time_valid_face: {time.time()-t_valid}")
            # check valid_class need to have 2 class0, 1 class1, 1 class2, 2 class3
            # using count valid_cls_inds
            count_cls = Counter(valid_cls_inds)
            # sort count_cls by key
            count_cls = dict(sorted(count_cls.items()))
            logger.info(f"----count_cls: {count_cls}")
            # if unique classes are not enough
            if len(count_cls) < 4:
                return {"success": False, "error_code": 8003, "error": "Don't find any user"}
            elif not (count_cls[0] >=2 and count_cls[1]>=1 and count_cls[2]>=1 and count_cls[3]>=2):
                return {"success": False, "error_code": 8003, "error": "Don't find any user"}
        return {"success": True}
        
    except Exception as e:
        tb_str = traceback.format_exc()
        LOGGER_APP.error(f"Traceback: {tb_str}")
        return {"success": False, "error_code": 8008, "error": str(e)}


@app.post("/api/registerFace")
async def register_face_v2(
    params: Person = Body(...),
    images: List[UploadFile] = File(...)
):
    try:
        if len(images) == 0:
            return {"success": False, "error_code": 8005, "error": "No users have been registered!"}
        logger.info(f"----params: {params}")
        code = params.code
        special_letters = set(code).difference(ALLOWED_CODE_CHARS)
        logger.info(f"----special_letters: {special_letters}")
        if special_letters:
            return {"success": False, "error_code": 8010, "error": "There are some special letters in user code!"}

        path_avatar = f"{IMG_AVATAR}/{code}/face_0.jpg"
        path_code = os.path.join(PATH_IMG_AVATAR, code)
        os.makedirs(path_code, exist_ok=True)

        name = params.name
        birthday = params.birthday
        imgs = []
        img_infor = []
        image_count = 1
        code_images = []
        for i, image in enumerate(images):
            code_image = "{}_{}".format(code, image_count)
            image_count += 1
            image_byte = await image.read()
            nparr = np.frombuffer(image_byte, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            img_infor.append(img.shape[:2])
            imgs.append(img)
            code_images.append(code_image)

        # imgs = np.array(imgs)
        img_infor = np.array(img_infor)
        t_det = time.time()
        results = facedet.inference(imgs)
        logger.info(f"---------time_facedet: {time.time()-t_det}")
        dets, miss_det, croped_image = results
        if len(croped_image) == 0:
            return {"success": False, "error_code": 8001, "error": "Don't find any face"}
        t_feature = time.time()
        feature = arcface.get_feature_without_det(croped_image)
        logger.info(f"---------time_feature: {time.time()-t_feature}")
        feature = np.array(feature, dtype=np.float16)
        # _ensure_collection()

        points = []
        for i, ft in enumerate(feature):
            # num_face = len(os.listdir(path_code))
            # cv2.imwrite(f"{path_code}/face_{num_face}.jpg", imgs[i])
            payload = {
                "code": code,
                "code_image": code_images[i],
                "name": name,
                "birthday": birthday,
                "avatar": path_avatar,
            }
            points.append(
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=ft.tolist(),
                    payload=payload,
                )
            )
        qdrantclient.upsert(collection_name=collection_name, points=points)
        return {"success": True}
    except Exception as e:
        tb_str = traceback.format_exc()
        LOGGER_APP.error(f"Traceback: {tb_str}")
        return {"success": False, "error_code": 8008, "error": str(e)}

@app.post("/api/registerSubFace")
async def register_Subface_v2(
    params: Person = Body(...),
    images: List[UploadFile] = File(...)
):
    try:
        if len(images) == 0:
            return {"success": False, "error_code": 8005, "error": "No users have been registered!"}
        logger.info(f"----params: {params}")
        code = params.code
        special_letters = set(code).difference(ALLOWED_CODE_CHARS)
        logger.info(f"----special_letters: {special_letters}")
        if special_letters:
            return {"success": False, "error_code": 8010, "error": "There are some special letters in user code!"}

        path_avatar = f"{IMG_AVATAR}/{code}/face_0.jpg"
        path_code = os.path.join(PATH_IMG_AVATAR, code)
        os.makedirs(path_code, exist_ok=True)

        name = params.name
        birthday = params.birthday
        imgs = []
        img_infor = []
        code_filter = models.Filter(
                must=[models.FieldCondition(key="code", match=models.MatchValue(value=code))]
            )
        images_by_code, _ = qdrantclient.scroll(
            collection_name=collection_name,
            scroll_filter=code_filter,
            with_payload=True,
            with_vectors=False,
            limit=100
        )


        code_images_sorted = [
            int(p.payload["code_image"].split('_')[-1])
            for p in images_by_code
            if "code_image" in p.payload
        ]

        image_count = max(code_images_sorted) + 1

        code_images = []
        for i, image in enumerate(images):
            code_image = "{}_{}".format(code, image_count)
            image_count += 1
            image_byte = await image.read()
            nparr = np.frombuffer(image_byte, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            img_infor.append(img.shape[:2])
            imgs.append(img)
            code_images.append(code_image)

        # imgs = np.array(imgs)
        img_infor = np.array(img_infor)
        t_det = time.time()
        results = facedet.inference(imgs)
        logger.info(f"---------time_facedet: {time.time()-t_det}")
        dets, miss_det, croped_image = results
        if len(croped_image) == 0:
            return {"success": False, "error_code": 8001, "error": "Don't find any face"}
        t_feature = time.time()
        feature = arcface.get_feature_without_det(croped_image)
        logger.info(f"---------time_feature: {time.time()-t_feature}")
        feature = np.array(feature, dtype=np.float16)
        # _ensure_collection()

        points = []
        for i, ft in enumerate(feature):
            # num_face = len(os.listdir(path_code))
            # cv2.imwrite(f"{path_code}/face_{num_face}.jpg", imgs[i])
            payload = {
                "code": code,
                "code_image": code_images[i],
                "name": name,
                "birthday": birthday,
                "avatar": path_avatar,
            }
            points.append(
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=ft.tolist(),
                    payload=payload,
                )
            )
        qdrantclient.upsert(collection_name=collection_name, points=points)
        return {"success": True}
    except Exception as e:
        tb_str = traceback.format_exc()
        LOGGER_APP.error(f"Traceback: {tb_str}")
        return {"success": False, "error_code": 8008, "error": str(e)}

@app.post("/api/searchUser")
async def search_user_v2(image: UploadFile = File(...)):
    try:
        id_faces = qdrantclient.count(
            collection_name=collection_name,
        )
        if id_faces.count == 0:
            return {"success": False, "error_code": 8000, "error": "Don't have any registered user"}
        image_byte = await image.read()
        nparr = np.frombuffer(image_byte, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        t_det = time.time()
        #---------------------------face det-------------------------
        results = facedet.inference([img])
        LOGGER_APP.info(f"---------time_facedet: {time.time()-t_det}")
        dets, miss_det, croped_image = results
        if len(croped_image)==0:
            return {"success": False, "error_code": 8001, "error": "Don't find any face"}
        t_valid = time.time()
        _, valid_scores, valid_cls_inds = yolovaliface.infer(croped_image)
        logger.info(f"---------time_valid_face: {time.time()-t_valid}")
        count_cls = Counter(valid_cls_inds)
            # sort count_cls by key
        count_cls = dict(sorted(count_cls.items()))
        logger.info(f"----count_cls: {count_cls}")
        # if unique classes are not enough
        if len(count_cls) < 4:
            return {"success": False, "error_code": 8003, "error": "Don't find any user"}
        elif not (count_cls[0] >=2 and count_cls[1]>=1 and count_cls[2]>=1 and count_cls[3]>=2):
            return {"success": False, "error_code": 8003, "error": "Don't find any user"}
        box = dets[0]["loc"]
        # print((box[2]-box[0])*(box[3]-box[1]))
        area_img = img.shape[0]*img.shape[1]
        w_crop = (box[2]-box[0])
        h_crop = (box[3]-box[1])
        # if not area_img*0.15<w_crop*h_crop<area_img*0.3:
        #     return {"success": False, "error_code": 8009, "error": "Face size is not true"}
        #---------------spoofing--------------
        box_expand = np.array([max(box[0]-w_crop,0), max(box[1]-h_crop,0), min(box[2]+w_crop, img.shape[1]), min(box[3]+h_crop, img.shape[0])], dtype=int)
        result = spoofingdet.inference([img[box_expand[1]:box_expand[3], box_expand[0]:box_expand[2]]])[0]
        # result = SPOOFINGDET.inference([img])[0]
        LOGGER_APP.info(f"---------result_spoofing: {result}")
        if result[1] > 0.78:
            # img_list = os.listdir(f"{PATH_IMG_SPOOFING}")
            # cv2.imwrite(f"{PATH_IMG_SPOOFING}/{len(img_list)}.jpg", img[box_expand[1]:box_expand[3], box_expand[0]:box_expand[2]])
            return {"success": False, "error_code": 8002, "error": "Fake face image"}
        #//////////////////////////////////////
        #////////////////////////////////////////////////////////////
        LOGGER_APP.info(f"------Duration det: {time.time()-t_det}")

        t_reg = time.time()
        #---------------------------face reg-------------------------
        feature = arcface.get_feature_without_det(croped_image).squeeze(0)
        feature = np.array(feature, dtype=np.float16)
        #////////////////////////////////////////////////////////////
        LOGGER_APP.info(f"------Duration reg: {time.time()-t_reg}")
        result = qdrantclient.query_points(
            collection_name= collection_name,
            query=feature, # <--- Dense vector
            search_params=models.SearchParams(hnsw_ef=128, exact=False),
            limit=1,
            with_payload=True,
        )                                                                                                                                      
        infor_face = None
        if len(result.points) > 0:
            best_point = result.points[0]
            similarity_best = best_point.score
            LOGGER_APP.info(similarity_best)
            if similarity_best < 0.99:
                infor_face = best_point.payload

                #save image to train
                path_user = f"{PATH_IMG_REC}/{infor_face['code']}"
                if not os.path.exists(path_user):
                    os.makedirs(path_user, exist_ok=True)
                img_list = os.listdir(f"{path_user}")
                box = box.astype(int)
                cv2.imwrite(f"{path_user}/{len(img_list)}.jpg", img[box[1]:box[3], box[0]:box[2]])
        #/////////////////////////////////////////////////////////////
        # LOGGER_APP.info(f"------Duration compare: {time.time()-t_comp}")
        if infor_face is None:
            return {"success": False, "error_code": 8003, "error": "Don't find any user"}
        LOGGER_APP.info(f"----infor_face; {infor_face}")
        return {"success": True, "information": {"code": infor_face['code'], "name": infor_face['name'], "birthday": infor_face['birthday'], "avatar": infor_face['avatar'], "similarity": float(similarity_best)}}
    except Exception as e:
        tb_str = traceback.format_exc()
        LOGGER_APP.error(f"Traceback: {tb_str}")
        return {"success": False, "error_code": 8008, "error": str(e)}


@app.post("/api/deleteUser")
async def delete_user_v2(codes: List[str] = ["001099008839"]):
    try:
        # _ensure_collection()

        missing_codes = []
        for code in codes:
            code_filter = models.Filter(
                must=[models.FieldCondition(key="code", match=models.MatchValue(value=code))]
            )
            count_result = qdrantclient.count(
                collection_name=collection_name,
                count_filter=code_filter,
                exact=True,
            )
            if count_result.count == 0:
                missing_codes.append(code)

        # if missing_codes:
        #     return {
        #         "success": False,
        #         "error_code": 8006,
        #         "error": f"User {tuple(missing_codes)} has not been registered!",
        #     }

        delete_filter = models.Filter(
            should=[
                models.FieldCondition(key="code", match=models.MatchValue(value=code))
                for code in codes
            ]
        )
        qdrantclient.delete(
            collection_name=collection_name,
            points_selector=models.FilterSelector(filter=delete_filter),
        )

        for code in codes:
            path_code = os.path.join(PATH_IMG_AVATAR, code)
            if os.path.exists(path_code):
                shutil.rmtree(path_code)

        return {"success": True}
    except Exception as e:
        tb_str = traceback.format_exc()
        LOGGER_APP.error(f"Traceback: {tb_str}")
        return {"success": False, "error_code": 8008, "error": str(e)}



@app.post("/api/getInformationUser")
async def get_information_user_v2(codes: List[str] = []):
    try:
        # _ensure_collection()
        infor_persons = {}
        LOGGER_APP.info(f"----codes: {codes}")

        if len(codes) == 0:
            next_page = None
            seen_codes = set()
            while True:
                points, next_page = qdrantclient.scroll(
                    collection_name=collection_name,
                    offset=next_page,
                    limit=256,
                    with_payload=True,
                )
                if not points:
                    break
                for pt in points:
                    payload = pt.payload or {}
                    code = payload.get("code")
                    if not code or code in seen_codes:
                        continue
                    seen_codes.add(code)
                    infor_persons[code] = {
                        "id": code,
                        "name": payload.get("name"),
                        "birthday": payload.get("birthday"),
                        "avatar": payload.get("avatar"),
                    }
                if next_page is None:
                    break
        else:
            for code in codes:
                code_filter = models.Filter(
                    must=[models.FieldCondition(key="code", match=models.MatchValue(value=code))]
                )
                points, _ = qdrantclient.scroll(
                    collection_name=collection_name,
                    scroll_filter=code_filter,
                    limit=1,
                    with_payload=True,
                )
                if not points:
                    infor_persons[code] = "No register"
                    continue
                payload = points[0].payload or {}
                infor_persons[code] = {
                    "id": payload.get("code", code),
                    "name": payload.get("name"),
                    "birthday": payload.get("birthday"),
                    "avatar": payload.get("avatar"),
                }

        return {"success": True, "information": infor_persons}
    except Exception as e:
        tb_str = traceback.format_exc()
        LOGGER_APP.error(f"Traceback: {tb_str}")
        return {"success": False, "error_code": 8008, "error": str(e)}


@app.post("/api/deleteAllUser")
async def delete_all_user_v2():
    try:
        qdrantclient.delete(
            collection_name=collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter()
            )
        )
        # _ensure_collection()
        if os.path.exists(PATH_IMG_AVATAR):
            shutil.rmtree(PATH_IMG_AVATAR)
            os.mkdir(PATH_IMG_AVATAR)
        return {"success": True}
    except Exception as e:
        tb_str = traceback.format_exc()
        LOGGER_APP.error(f"Traceback: {tb_str}")
        return {"success": False, "error_code": 8008, "error": str(e)}


@app.post("/api/checkFailFacev2")
async def check_fail_face_v2():
    pass


@app.post("/api/deleteFailFacev2")
async def delete_fail_face_v2():
    pass




async def register_from_json_controller(
    json_path: str = None,
    images_root: str = None,
    limit: int = 0,
    api_path: str = "/api/registerFace",
):
    """Batch register users from `fi/result.json` using internal ASGI calls.

    Input records must follow this schema:
    {"code": "...", "name": "...", "images": ["img1.jpg", "img2.png", ...]}
    """
    import json as _json
    import mimetypes
    import re
    from pathlib import Path
    from httpx import AsyncClient, ASGITransport

    def _resolve_path(path_value: str, default_path: Path, base_path: Path) -> Path:
        if not path_value:
            return default_path
        raw_path = Path(path_value)
        if raw_path.is_absolute():
            return raw_path
        cwd_candidate = (Path.cwd() / raw_path)
        if cwd_candidate.exists():
            return cwd_candidate
        return base_path / raw_path

    def _normalize_code(raw_code: str) -> str:
        return (raw_code or "").strip()

    def _extract_birthday(normalized_code: str) -> str:
        match = re.search(r"_([0-9]{8})$", normalized_code or "")
        if not match:
            return "01/01/1999"
        birthday_raw = match.group(1)
        dd, mm, yyyy = birthday_raw[0:2], birthday_raw[2:4], birthday_raw[4:8]
        return f"{dd}/{mm}/{yyyy}"

    try:
        project_root = Path(__file__).resolve().parents[1]
        default_json_path = project_root / "fi" / "result.json"
        default_images_root = project_root / "fi" / "images"

        jp = _resolve_path(json_path, default_json_path, project_root)
        if not jp.exists():
            return {"success": False, "error": f"json file not found: {jp}"}

        image_root_path = _resolve_path(images_root, default_images_root, project_root)
        if not image_root_path.exists():
            return {"success": False, "error": f"images root not found: {image_root_path}"}

        with open(jp, "r", encoding="utf-8") as fh:
            records = _json.load(fh)
        if not isinstance(records, list):
            return {"success": False, "error": "json format is invalid: expected a list"}

        if limit and limit > 0:
            records = records[:limit]

        summary = {
            "total": len(records),
            "registered": 0,
            "failed": 0,
            "normalized_codes": {},
            "results": [],
        }

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://batch.local") as client:
            for rec in records:
                original_code = str(rec.get("code") or "").strip()
                name = str(rec.get("name") or "").strip()
                image_names = rec.get("images")
                entry = {
                    "original_code": original_code,
                    "normalized_code": "",
                    "name": name,
                    "image_count": len(image_names) if isinstance(image_names, list) else 0,
                    "status": "failed",
                }

                if not original_code:
                    entry["error"] = "missing code"
                    summary["failed"] += 1
                    summary["results"].append(entry)
                    continue

                if not isinstance(image_names, list) or len(image_names) == 0:
                    entry["error"] = "missing images[]"
                    summary["failed"] += 1
                    summary["results"].append(entry)
                    continue

                normalized_code = _normalize_code(original_code)
                entry["normalized_code"] = normalized_code
                if normalized_code != original_code:
                    summary["normalized_codes"][original_code] = normalized_code

                if not normalized_code:
                    entry["error"] = "normalized code is empty"
                    summary["failed"] += 1
                    summary["results"].append(entry)
                    continue

                birthday = _extract_birthday(normalized_code)

                missing_images = []
                image_paths = []
                for image_name in image_names:
                    if not isinstance(image_name, str) or not image_name.strip():
                        missing_images.append(str(image_name))
                        continue
                    image_path = image_root_path / image_name.strip()
                    if not image_path.exists():
                        missing_images.append(str(image_path))
                        continue
                    image_paths.append(image_path)

                if missing_images:
                    entry["error"] = "missing image files"
                    entry["missing_images"] = missing_images
                    summary["failed"] += 1
                    summary["results"].append(entry)
                    continue

                file_handles = []
                try:
                    files = []
                    for image_path in image_paths:
                        file_handle = open(image_path, "rb")
                        file_handles.append(file_handle)
                        mime = mimetypes.guess_type(str(image_path))[0] or "application/octet-stream"
                        files.append(("images", (image_path.name, file_handle, mime)))

                    params_payload = {"code": normalized_code, "name": name, "birthday": birthday}
                    data = {"params": _json.dumps(params_payload, ensure_ascii=False)}
                    resp = await client.post(api_path, data=data, files=files, timeout=300.0)

                    try:
                        response_payload = resp.json()
                    except Exception:
                        response_payload = {"raw": resp.text}

                    entry["response"] = response_payload
                    if resp.status_code == 200 and isinstance(response_payload, dict) and response_payload.get("success"):
                        entry["status"] = "registered"
                        summary["registered"] += 1
                    else:
                        entry["error"] = f"registration failed (status_code={resp.status_code})"
                        summary["failed"] += 1
                except Exception as ex:
                    entry["error"] = str(ex)
                    summary["failed"] += 1
                finally:
                    for file_handle in file_handles:
                        try:
                            file_handle.close()
                        except Exception:
                            pass

                summary["results"].append(entry)

        return {"success": summary["failed"] == 0, "result": summary}
    except Exception as e:
        tb_str = traceback.format_exc()
        LOGGER_APP.error(f"Traceback: {tb_str}")
        return {"success": False, "error_code": 8008, "error": str(e)}


def register_from_json_controller_sync(
    json_path: str = None,
    images_root: str = None,
    limit: int = 0,
    api_path: str = "/api/registerFace",
):
    """Synchronous wrapper around `register_from_json_controller` for scripts/CLI."""
    import asyncio
    return asyncio.run(
        register_from_json_controller(
            json_path=json_path,
            images_root=images_root,
            limit=limit,
            api_path=api_path,
        )
    )


if __name__=="__main__":
	host = "0.0.0.0"
	port = 8010

	uvicorn.run("controller:app", host=host, port=port, log_level="info", reload=True)



"""
8000: "Don't have any registered user"
8001: "Don't find any face"
8002: "Fake face image"
8003: "Don't find any user"
8004: "This user has been registered!"
8005: "No users have been registered!"
8006: "This user has not been registered!"
8007: "Too many faces in this image"
8008: error system
8009: "Face size is not true"
8010: "There are some special letters in user code!"
8011: "Lost some face parts, please register again with clear face image"
"""
