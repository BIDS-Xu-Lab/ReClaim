import numpy as np
import torch
from tqdm import tqdm


def get_visit_location(seq_list):
    visit_location = []
    for i, token in enumerate(seq_list):
        if token.startswith("<ATT-"):
            visit_location.append(i)
    return visit_location


def get_disease_points_ages_and_locations_by_visit(
    tokens,
    demo_end_index=9,
    age_index=3,
    max_length=4096,
    first_occurrence=True,
    att_type="day",
):
    """
    Extract disease points and ages from a sequence and their visit sequence positions.
    """
    disease_points = []
    disease_points_ages = []
    disease_points_visit_location = []
    start_day_age = int(tokens[age_index][5:-1]) * 365.25
    passed_years = 0
    passed_day = 0
    disease_token = set()
    if att_type == "day":
        att_gap = 1
    elif att_type == "week":
        att_gap = 7
    elif att_type == "month":
        att_gap = 30
    else:
        raise ValueError(f"Invalid att_type: {att_type}")

    visit_location = get_visit_location(tokens)
    visit_location.append(len(tokens) - 1)
    visit_idx = 0

    i = demo_end_index
    length = min(len(tokens), max_length)
    while i < length:
        current_token = tokens[i]
        if current_token.startswith("<NY"):
            passed_years += 1
            passed_day = 0

        if current_token.startswith("<ATT-"):
            visit_idx += 1
            att_value = int(current_token[5:-1]) if current_token != "<ATT-0>" else 0
            passed_day += att_gap * att_value
        elif current_token.startswith("<DX-MAJOR"):
            if first_occurrence:
                if current_token not in disease_token:
                    disease_points.append(i)
                    disease_points_ages.append(
                        start_day_age + passed_years * 365.25 + passed_day
                    )
                    disease_token.add(current_token)
                    try:
                        disease_points_visit_location.append(visit_location[visit_idx])
                    except:
                        print(f"visit_idx: {visit_idx}, visit_location: {visit_location}, tokens: {tokens}")
                        raise ValueError(f"visit_idx: {visit_idx}, visit_location: {visit_location}, tokens: {tokens}")
            else:
                disease_points.append(i)
                disease_points_ages.append(
                    start_day_age + passed_years * 365.25 + passed_day
                )
                disease_points_visit_location.append(visit_location[visit_idx])
        i += 1

    return disease_points, disease_points_ages, disease_points_visit_location


def get_absolute_disease_points_ages_and_locations_by_visit(
    tokens,
    demo_end_index=9,
    age_index=3,
    max_length=4096,
    first_occurrence=True,
    att_type="day",
):
    """
    Extract disease points and ages using absolute AGE/ATT tokens.
    """
    disease_points = []
    disease_points_ages = []
    disease_points_visit_location = []
    current_age = int(tokens[age_index][5:-1]) * 365.25
    current_day = 0
    disease_token = set()
    if att_type == "day":
        att_gap = 1
    elif att_type == "week":
        att_gap = 7
    elif att_type == "month":
        att_gap = 30
    else:
        raise ValueError(f"Invalid att_type: {att_type}")

    visit_location = get_visit_location(tokens)
    visit_location.append(len(tokens) - 1)
    visit_idx = 0

    i = demo_end_index
    length = min(len(tokens), max_length)
    while i < length:
        current_token = tokens[i]
        if current_token.startswith("<AGE"):
            current_age = int(current_token[5:-1]) * 365.25
        elif current_token.startswith("<ATT-"):
            visit_idx += 1
            att_value = int(current_token[5:-1])
            current_day = att_gap * att_value
        elif current_token.startswith("<DX-MAJOR"):
            if first_occurrence:
                if current_token not in disease_token:
                    disease_points.append(i)
                    disease_points_ages.append(current_age + current_day)
                    disease_token.add(current_token)
                    try:
                        disease_points_visit_location.append(visit_location[visit_idx])
                    except:
                        print(f"visit_idx: {visit_idx}, visit_location: {visit_location}, tokens: {tokens}")
                        raise ValueError(f"visit_idx: {visit_idx}, visit_location: {visit_location}, tokens: {tokens}")
            else:
                disease_points.append(i)
                disease_points_ages.append(current_age + current_day)
                disease_points_visit_location.append(visit_location[visit_idx])
        i += 1

    return disease_points, disease_points_ages, disease_points_visit_location


def get_major_disease_id_map(tokenizer):
    """Get mapping of major disease tokens to their token IDs."""
    major_disease_ids_map = {}
    for token, idx in tokenizer.get_vocab().items():
        if token.startswith("<DX-MAJOR_"):
            major_disease_ids_map[token] = idx
    major_disease_ids_map = dict(
        sorted(major_disease_ids_map.items(), key=lambda item: item[1])
    )
    return major_disease_ids_map


def add_disease_points_ages_and_locations(
    data, demo_end_index=9, age_index=3, seq_len=4096, att_type="day", min_seq_len=10, absolute=False, longitudinal=False, first_occurrence=True
):
    """Process data to extract disease points and visit locations."""
    print("Extracting prediction points...")
    disease_points_all = []
    disease_points_ages_all = []
    disease_points_visit_location_all = []
    eval_seqs = []
    seqs_to_keep = []

    for i, item in tqdm(data.iterrows(), desc="Processing sequences"):
        if longitudinal:
            token_idx = item["total_token_count"] + 1
            num_prediction_tokens = item["token_count_thru_2022"] + 1
        else:
            token_idx = item["token_count_thru_2022"] + 1
            num_prediction_tokens = token_idx
        if num_prediction_tokens < min_seq_len:
            disease_points_all.append([])
            disease_points_ages_all.append([])
            disease_points_visit_location_all.append([])
            seqs_to_keep.append(False)
            eval_seqs.append([])
            continue
        tokens = item["seq"].split(" ")[:token_idx]
        eval_seqs.append(" ".join(tokens))
        if absolute:
            disease_point, disease_point_ages, disease_point_visit_location = (
                get_absolute_disease_points_ages_and_locations_by_visit(
                    tokens,
                    demo_end_index=demo_end_index,
                    age_index=age_index,
                    max_length=seq_len,
                    att_type=att_type,
                    first_occurrence=first_occurrence,
                )
            )
        else:
            disease_point, disease_point_ages, disease_point_visit_location = (
                get_disease_points_ages_and_locations_by_visit(
                    tokens,
                    demo_end_index=demo_end_index,
                    age_index=age_index,
                    max_length=seq_len,
                    att_type=att_type,
                    first_occurrence=first_occurrence,
                )
            )
        if len(disease_point) < 2:
            disease_points_all.append([])
            disease_points_ages_all.append([])
            disease_points_visit_location_all.append([])
            seqs_to_keep.append(False)
        else:
            disease_points_all.append(disease_point)
            disease_points_ages_all.append(disease_point_ages)
            disease_points_visit_location_all.append(disease_point_visit_location)
            seqs_to_keep.append(True)

    data = data[seqs_to_keep].reset_index(drop=True)
    data["disease_points"] = [
        dp for dp, keep in zip(disease_points_all, seqs_to_keep) if keep
    ]
    data["disease_points_ages"] = [
        dpa for dpa, keep in zip(disease_points_ages_all, seqs_to_keep) if keep
    ]
    data["disease_points_visit_location"] = [
        dpl
        for dpl, keep in zip(disease_points_visit_location_all, seqs_to_keep)
        if keep
    ]
    data["eval_seq"] = [
        new_seq for new_seq, keep in zip(eval_seqs, seqs_to_keep) if keep
    ]

    print(f"Kept {len(data)} sequences with at least 2 disease points")
    return data


def get_first_true_locations(mat):
    """
    For a 2-d boolean numpy array `mat`,
    return two numpy arrays: row_ids, col_ids, corresponding to the *first* True for each row (if any).
    """
    any_true = mat.any(axis=1)
    first_true_idx = mat.argmax(axis=1)
    row_ids = np.nonzero(any_true)[0]
    col_ids = first_true_idx[any_true]
    return row_ids, col_ids


def prepare_eval_inputs(dataloader, tokenizer, sex_index=1, sed_token_id=301):
    all_disease_tokens = []
    all_ages = []
    all_seq_points = []
    all_sex = []
    for batch in tqdm(dataloader):
        batch_data = (
            batch["tokens"],
            batch["disease_points"],
            batch["disease_points_ages"],
            batch["disease_points_visit_location"],
            batch["patient_ids"],
        )
        sexs = (
            (batch_data[0]["input_ids"][:, sex_index] == sed_token_id).cpu().numpy()
        )
        disease_ids = batch_data[0]["input_ids"].gather(
            1, torch.tensor(batch_data[1], dtype=torch.long)
        )
        disease_ids[batch_data[1] == -1] = tokenizer.pad_token_id
        ages = batch_data[2]
        all_disease_tokens.extend(disease_ids.tolist())
        all_ages.extend(ages.tolist())
        all_seq_points.extend(batch_data[3].tolist())
        all_sex.extend(sexs)
    all_disease_tokens = np.stack(all_disease_tokens)
    all_ages = np.stack(all_ages)
    all_seq_points = np.stack(all_seq_points)
    all_sex = np.stack(all_sex)
    print(all_disease_tokens.shape, all_ages.shape, all_seq_points.shape)
    return all_disease_tokens, all_ages, all_seq_points, all_sex


def keep_row_first_true(mat):
    """
    For each row, set the first True value to 1, and the rest to 0. If no True values, row remains zeros.
    """
    any_true = mat.any(axis=1)
    first_idx = mat.argmax(axis=1)
    new_mat = np.zeros_like(mat)
    row_indices = np.where(any_true)[0]
    new_mat[row_indices, first_idx[row_indices]] = 1
    return new_mat


def _precompute_disease_case_control_shared(
    all_disease_tokens,
    all_ages,
    all_seq_points,
    all_sex,
    age_groups,
    offset=365.25,
    pad_token_id=2,
):
    """
    Precompute all disease-invariant arrays for case-control matching.
    Call once before processing multiple diseases to avoid redundant O(n*m^2) work.
    """
    age_groups = np.asarray(age_groups)
    age_step = age_groups[1] - age_groups[0] if len(age_groups) > 1 else 5
    sex_masks = {"male": all_sex, "female": ~all_sex}
    inds = np.arange(all_disease_tokens.shape[0])

    precomputed = {}
    for sex in ["male", "female"]:
        mask = sex_masks[sex]
        new_inds = inds[mask]
        disease_ids = all_disease_tokens[mask]
        ages = all_ages[mask]
        seq_points = all_seq_points[mask]

        pred_idx_precompute = (
            ages[:, :-1][:, :, np.newaxis] < ages[:, 1:][:, np.newaxis, :] - offset
        ).sum(1) - 1
        pred_idx_precompute[disease_ids[:, :-1] == pad_token_id] = -1

        pred_ages = np.empty_like(pred_idx_precompute, dtype=np.float64)
        row_idx = np.arange(pred_idx_precompute.shape[0])[:, None]
        pred_ages[:] = ages[row_idx, pred_idx_precompute]
        pred_ages[pred_idx_precompute == -1] = 1000 * 365.25

        precomputed[sex] = {
            "new_inds": new_inds,
            "disease_ids": disease_ids,
            "ages": ages,
            "seq_points": seq_points,
            "pred_idx_precompute": pred_idx_precompute,
            "pred_ages": pred_ages,
            "age_step": age_step,
        }
    precomputed["age_groups"] = age_groups
    return precomputed


def get_disease_case_control_ids_from_precomputed(
    disease_token,
    precomputed,
):
    """
    Get case-control IDs for a single disease using precomputed shared data.
    """
    age_groups = precomputed["age_groups"]
    all_out = []
    for sex in ["male", "female"]:
        pc = precomputed[sex]
        new_inds = pc["new_inds"]
        disease_ids = pc["disease_ids"]
        seq_points = pc["seq_points"]
        pred_idx_precompute = pc["pred_idx_precompute"]
        pred_ages = pc["pred_ages"]
        age_step = pc["age_step"]

        for age in age_groups:
            age_mask = np.logical_and(
                pred_ages / 365.25 >= age, pred_ages / 365.25 < age + age_step
            ) & (pred_idx_precompute != -1)
            if age_mask.sum() < 2:
                continue

            case_mask = disease_ids[:, 1:] == disease_token
            case_mask = keep_row_first_true(case_mask)
            control_mask = (disease_ids[:, 1:] != disease_token) * (
                ~((disease_ids[:, 1:] == disease_token).any(-1))
            )[..., None]

            case_ids = get_first_true_locations(age_mask & case_mask)
            control_ids = get_first_true_locations(age_mask & control_mask)
            if len(case_ids[0]) == 0 or len(control_ids[0]) == 0:
                continue

            case_pred_ids = pred_idx_precompute[case_ids]
            control_pred_ids = pred_idx_precompute[control_ids]

            case_seq_points = seq_points[case_ids[0], case_pred_ids]
            control_seq_points = seq_points[control_ids[0], control_pred_ids]

            out = {
                "sex": sex,
                "disease_token": disease_token,
                "age": age,
                "age_step": age_step,
                "case_ids": new_inds[case_ids[0]],
                "control_ids": new_inds[control_ids[0]],
                "case_seq_points": case_seq_points,
                "control_seq_points": control_seq_points,
            }
            all_out.append(out)
    return all_out
