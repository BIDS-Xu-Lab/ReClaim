import random

def get_first_year_location(seq_list):
    year_location = [] ## start from the first token
    for i, token in enumerate(seq_list):
        if token.startswith("<NY"):
            year_location.append(i)
    return year_location

def get_att_visit_location(seq_list):
    visit_location = []  ## start from the first token
    for i, token in enumerate(seq_list):
        if token.startswith("<ATT-"):
            visit_location.append(i)
    return visit_location

def get_instruct_seq_dx(
    seq,
    demo_end_index=3,
    split_by="visit",
    instruct_token="<INSTRUCT-DX>",
    instruct_position="before",
    disease_set=None
    ):
    """
    generate the instruction sequence for the disease prediction task

    demo_end_index: the index of the demo end token
    split the squence after the demo end token
    before the split point, all tokens are kept
    after the split point, only the major disease tokens are kept
    split_by_visit: if True, split the sequence by the visit location, otherwise split the sequence by a random point
    intruct_position: 'before' means after the demo end token or 'after' means before the split point
    """
    tokens = seq.split(" ")
    tokens = ["<sos>"] + tokens #! modification: add <sos> token to the beginning of the sequence
    if split_by == "visit":
        visit_location = get_att_visit_location(tokens)
        assert (
            visit_location[0] > demo_end_index 
        ), f"demo end token is not the first token, visit_location: {visit_location}, demo_end_index: {demo_end_index}, tokens: {tokens}"
        split_point = random.choice(visit_location)
    elif split_by == "first_year":
        year_location = get_first_year_location(tokens)
        assert (
            year_location[0] > demo_end_index 
        ), f"demo end token is not the first token, year_location: {year_location}, demo_end_index: {demo_end_index}, tokens: {tokens}"
        split_point = year_location[0]
    elif split_by == "demo":
        split_point = demo_end_index + 1
    elif split_by == "random":
        split_point = random.randint(demo_end_index, len(tokens) - 1)    
    else:
        raise ValueError(f"Invalid split_by: {split_by}")

    if instruct_position == "before":
        instruct_seq = (
            tokens[: demo_end_index + 1]
            + [instruct_token]
            + tokens[demo_end_index + 1 : split_point]
        )
    elif instruct_position == "after":
        instruct_seq = tokens[:split_point] + [instruct_token]
    else:
        raise ValueError(f"Invalid instruct_position: {instruct_position}")

    if isinstance(disease_set, list):
        disease_set = set(disease_set)

    ### get disease before the split point
    disease_token = set()
    disease_after_split_point = []
    for i in range(demo_end_index + 1, split_point):
        if tokens[i].startswith("<DX-MAJOR"):
            disease_token.add(tokens[i])

    for i in range(split_point, len(tokens)):
        token = tokens[i]

        if token.startswith("<DX-MAJOR") and token not in disease_token:
            disease_token.add(tokens[i])
            major_code = token.split("_")[1].rstrip(">")

            if major_code in disease_set:
                disease_after_split_point.append(tokens[i])
                instruct_seq.append(tokens[i])

    if len(disease_after_split_point) == 0: #! modification: if no disease after the split point, return None
        return None

    instruct_seq.append("<eos>")

    instruct_seq = " ".join(instruct_seq)
    return instruct_seq, split_point