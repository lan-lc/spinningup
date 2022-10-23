import torch
import os

def rename_models(rootdir):
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            # print (os.path.join(subdir, file), file)
            filepath = subdir + os.sep + file
            if file.startswith("model") and file.endswith(".pt") and not file.endswith("_13.pt"):
                print (filepath)
                state_dict = torch.load(filepath, map_location="cpu")
                # print(subdir + os.sep + file[:-3] + "_13.pt")
                torch.save(state_dict, subdir + os.sep + file[:-3] + "_13.pt", _use_new_zipfile_serialization=False)



# rename_models('./data/ttn_2/')
# rename_models('./data/gsac_cs/')
# rename_models('./data/test_algo/')
# rename_models('./data/gsac_cs_2/')
# rename_models('./data/ttn_3/')
# rename_models('./data/gsac3_cs/')
# rename_models('./data/gsac4/')
#rename_models('./data/gsac4_tttr75/')
rename_models('./data/gsac6/')
