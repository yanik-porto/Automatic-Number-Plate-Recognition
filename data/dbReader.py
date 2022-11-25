def readAnnot(annotPath):
    entries = []
    with open(annotPath, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            w = line.split(" ")
            entries.append(w)
    return entries

def readAnnotAsRectAndGt(annotPath):
    entries = readAnnot(annotPath)
    rectAndGt = []
    for entry in entries:
        if len(entry) < 6:
            print("less than 6 entries in rct file : ", annotPath)
            continue

        gt = entry[5]

        # w = filter(None, w)
        rectStr = entry[1:5]
        rect = [int(string) for string in rectStr]
        rectAndGt.append((rect, gt))
    return rectAndGt