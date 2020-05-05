import xmltodict, json, os, re, sys
import numpy as np

dialogues = []

def process_file(id, filename_prefix):
    print(filename_prefix)
    
    f_annotation = open("%s.aa" % filename_prefix)
    annotations = xmltodict.parse(''.join(f_annotation.readlines()))["annotations"]
    units = annotations["unit"]
    if not 'relation' in annotations:
        relations = []
    else:
        relations = annotations["relation"]
    schema = annotations["schema"] if 'schema' in annotations else []

    f_discourse = open("%s.ac" % filename_prefix)
    discourse = f_discourse.readline()
    for i in range(len(discourse)):
        if ord(discourse[i]) >= 128: discourse = discourse[:i] + " " + discourse[i+1:]
    
    edus, buf_dialogues = {}, {}
    
    for item in units:
        _id = item["@id"]
        start = int(item["positioning"]["start"]["singlePosition"]["@index"])
        end = int(item["positioning"]["end"]["singlePosition"]["@index"])
        _type = item["characterisation"]["type"]
        if _type in ["Turn", "NonplayerTurn"]: continue
        elif _type == "Dialogue":
            buf_dialogues[_id] = {
                "start": start,
                "end": end,
                "edus": {},
                "cdus": {},
                "relations": []
            }
        else:
            edus[_id] = {
                "id": _id,
                "type": _type,
                "text": discourse[start:end],
                "start": start,
                "end": end
            }

    belong_to = {}
    for id_edu in edus:
        edu = edus[id_edu]
        found = False
        for id_dialogue in buf_dialogues:
            dialog = buf_dialogues[id_dialogue]
            if dialog["start"] <= edu["start"] and dialog["end"] >= edu["end"]:
                found = True
                dialog["edus"][id_edu] = edu
                belong_to[id_edu] = id_dialogue
                break
        if not found:
            raise Warning("Dialogue not found")
    
    if type(schema) != list: schema = [schema] 
    for item in schema:
        _id = item["@id"]
        _type = item["characterisation"]["type"]
        if item["positioning"] == None: continue
        
        cdu = []
        if "embedded-unit" in item["positioning"]:
            if type(item["positioning"]["embedded-unit"]) == list:
                cdu = [unit["@id"] for unit in item["positioning"]["embedded-unit"]]
            else:
                cdu = [item["positioning"]["embedded-unit"]["@id"]]
            for edu in cdu:
                if not edu in edus:
                    cdu.remove(edu)
        if "embedded-schema" in item["positioning"]:
            if type(item["positioning"]["embedded-schema"]) == list:
                cdu += [unit["@id"] for unit in item["positioning"]["embedded-schema"]]
            else:
                cdu += [item["positioning"]["embedded-schema"]["@id"]]
        belong_to[_id] = belong_to[cdu[0]]
        buf_dialogues[belong_to[_id]]["cdus"][_id] = cdu
        
    if type(relations) != list: relations = [relations]
    for item in relations:
        _id = item["@id"]
        x = item["positioning"]["term"][0]["@id"]
        y = item["positioning"]["term"][1]["@id"]
        _type = item["characterisation"]["type"]
        buf_dialogues[belong_to[x]]["relations"].append({
            "type": _type,
            "x": x,
            "y": y
        })
        
    for _id in buf_dialogues:
        buf_dialogues[_id]["id"] = id
        dialogues.append(buf_dialogues[_id])
        
def process_dialogue(dialogue):
    has_incoming = {}
    
    for relation in dialogue["relations"]:
        has_incoming[relation["y"]] = True
       
    for _id in dialogue["edus"]:
        edu = dialogue["edus"][_id]
        if edu["type"] == "paragraph": continue
        
        for _id_para in dialogue["edus"]:
            def parse_speaker(text):
                return (text.split())[2]
            
            para = dialogue["edus"][_id_para]
            if para["type"] != "paragraph": continue
            if para["start"] <= edu["start"] and para["end"] >= edu["end"]:
                edu["speaker"] = parse_speaker(para["text"])
    
    idx = {}
    dialogue["edu_list"] = []
    for _id in dialogue["edus"]:
        if dialogue["edus"][_id]["type"] != "paragraph":
            dialogue["edu_list"].append(dialogue["edus"][_id])
    dialogue["edu_list"] = sorted(dialogue["edu_list"], key=lambda edu: edu["start"])
    for i in range(len(dialogue["edu_list"])):
        edu = dialogue["edu_list"][i]
        idx[edu["id"]] = i
        
    for i, edu in enumerate(dialogue["edu_list"]):
        print(i, edu["speaker"], ":", edu["text"])
       
    print("===")

    for relation in dialogue["relations"]:
        def get_head(x):
            if x in dialogue["edus"]: return x
            else: 
                for du in dialogue["cdus"][x]:
                    if not du in has_incoming: return get_head(du)
                raise Warning("Can't find the recursive head")
            
        relation["x"] = idx[get_head(relation["x"])]
        relation["y"] = idx[get_head(relation["y"])]
        
    dialogue_cleaned = {
        "id": dialogue["id"],
        "edus": [],
        "relations": []
    }
    
    for edu in dialogue["edu_list"]:
        dialogue_cleaned["edus"].append({
            "speaker": edu["speaker"],
            "text": edu["text"]
        })
    for relation in dialogue["relations"]:
        dialogue_cleaned["relations"].append({
            "type": relation["type"],
            "x": relation["x"],
            "y": relation["y"]
        })
        
    return dialogue_cleaned
      
input_dir = sys.argv[1]
output_file = sys.argv[2]
      
dirs = os.listdir(input_dir)
for directory in dirs:
    path = os.path.join(os.path.join(input_dir, directory), "discourse/GOLD")
    if os.path.exists(path):
        for filename in os.listdir(path):
            if re.match("\S*.ac", filename):
                id = filename[:filename.find('_')]
                process_file(id, os.path.join(path, filename[:filename.index(".")]))

dialogues_cleaned = []
retained = []
for dialogue in dialogues:
    dialog = process_dialogue(dialogue)
    dialogues_cleaned.append(dialog)
fout = open(output_file, "w")
fout.write(json.dumps(dialogues_cleaned))
fout.close()
print("%d dialogues" % len(dialogues_cleaned))
