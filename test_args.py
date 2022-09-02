import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--name_id', type=int)
args =  parser.parse_args()
if args.name_id == None:
    print('no name id')
    args.name_id = 10

print(args.name_id)

