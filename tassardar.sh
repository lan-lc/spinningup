python launcher_gsac.py --env Humanoid-v3 -sn 1 --num_runs 5 &
python launcher_gsac.py --env Humanoid-v3 -sn 2 -sr 1 --num_runs 5 &
python launcher_gsac.py --env Humanoid-v3 -sn 2 -sr 0 --num_runs 5 &

sleep 30m

python launcher_gsac.py --env Walker2d-v3 -sn 1 --num_runs 5 &
python launcher_gsac.py --env Hopper-v3   -sn 1 --num_runs 5 &
python launcher_gsac.py --env Walker2d-v3 -sn 2 -sr 1 --num_runs 5 &

sleep 20m

python launcher_gsac.py --env Hopper-v3   -sn 2 -sr 1 --num_runs 5 &
python launcher_gsac.py --env Walker2d-v3 -sn 2 -sr 0 --num_runs 5 &
python launcher_gsac.py --env Hopper-v3   -sn 2 -sr 0 --num_runs 5 &



# velle
python launcher_gsac.py --env Humanoid-v3 -sn 5 -sr 1 --num_runs 5 &
python launcher_gsac.py --env Humanoid-v3 -sn 5 -sr 0 --num_runs 5 &
python launcher_gsac.py --env Humanoid-v3 -sn 10 -sr 1 --num_runs 5 &
python launcher_gsac.py --env Walker2d-v3 -sn 5 -sr 1 --num_runs 5 &

#tassardar
python launcher_gsac.py --env Hopper-v3   -sn 5 -sr 1 --num_runs 5 &
python launcher_gsac.py --env Walker2d-v3 -sn 5 -sr 0 --num_runs 5 &
python launcher_gsac.py --env Hopper-v3   -sn 5 -sr 0 --num_runs 5 &
python launcher_gsac.py --env Walker2d-v3 -sn 10 -sr 1 --num_runs 5 &
python launcher_gsac.py --env Hopper-v3   -sn 10 -sr 1 --num_runs 5 &



python launcher_gsac.py --env Walker2d-v3 -sn 10 -sr 0 --num_runs 5 &
python launcher_gsac.py --env Humanoid-v3 -sn 10 -sr 0 --num_runs 5 &
python launcher_gsac.py --env Hopper-v3   -sn 10 -sr 0 --num_runs 5 &


python launcher_gsac.py --env Humanoid-v3 -sn 10 -sr 0 --num_runs 5


#zetural

python launcher_gsac.py --env Humanoid-v3 -sn 1 --num_runs 5 -cs 50 &
python launcher_gsac.py --env Humanoid-v3 -sn 1 --num_runs 5 -cs 200 &
python launcher_gsac.py --env Humanoid-v3 -sn 1 --num_runs 5 -cs 500 &

python launcher_gsac.py --env Walker2d-v3 -sn 1 --num_runs 5 -cs 50  && python launcher_gsac.py --env Hopper-v3 -sn 1 --num_runs 5 -cs 50 & 
python launcher_gsac.py --env Walker2d-v3 -sn 1 --num_runs 5 -cs 200 && python launcher_gsac.py --env Hopper-v3 -sn 1 --num_runs 5 -cs 200 & 
python launcher_gsac.py --env Walker2d-v3 -sn 1 --num_runs 5 -cs 500 && python launcher_gsac.py --env Hopper-v3 -sn 1 --num_runs 5 -cs 500 & 


#tassardar
python launcher_gsac.py --env Humanoid-v3 -sn 1 --num_runs 5 -ttn 1 &
python launcher_gsac.py --env Humanoid-v3 -sn 1 --num_runs 5 -ttn 2 &
python launcher_gsac.py --env Humanoid-v3 -sn 1 --num_runs 5 -ttn 4 &
python launcher_gsac.py --env Humanoid-v3 -sn 1 --num_runs 5 -ttn 8 &

python launcher_gsac.py --env Walker2d-v3 -sn 1 --num_runs 5 -ttn 1 && python launcher_gsac.py --env Hopper-v3 -sn 1 --num_runs 5 -ttn 1 & 
python launcher_gsac.py --env Walker2d-v3 -sn 1 --num_runs 5 -ttn 2 && python launcher_gsac.py --env Hopper-v3 -sn 1 --num_runs 5 -ttn 2 & 
python launcher_gsac.py --env Walker2d-v3 -sn 1 --num_runs 5 -ttn 4 && python launcher_gsac.py --env Hopper-v3 -sn 1 --num_runs 5 -ttn 4 & 
python launcher_gsac.py --env Walker2d-v3 -sn 1 --num_runs 5 -ttn 8 && python launcher_gsac.py --env Hopper-v3 -sn 1 --num_runs 5 -ttn 8 & 


#zetural
python launcher_gsac.py --env Humanoid-v3 --num_runs 5 -tttr 0.33 &
python launcher_gsac.py --env Humanoid-v3 --num_runs 5 -tttr 0.2  &
python launcher_gsac.py --env Humanoid-v3 --num_runs 5 -tttr 0.1  &

sleep 10m

python launcher_gsac.py --env Walker2d-v3 --num_runs 5 -tttr 0.33 && python launcher_gsac.py --env Hopper-v3 --num_runs 5 -tttr 0.33 & 
python launcher_gsac.py --env Walker2d-v3 --num_runs 5 -tttr 0.2  && python launcher_gsac.py --env Hopper-v3 --num_runs 5 -tttr 0.2  & 
python launcher_gsac.py --env Walker2d-v3 --num_runs 5 -tttr 0.1  && python launcher_gsac.py --env Hopper-v3 --num_runs 5 -tttr 0.1  & 

#vella

python launcher_gsac.py --env Humanoid-v3 --num_runs 5 -tsr 0.25 &
python launcher_gsac.py --env Humanoid-v3 --num_runs 5 -tsr 0.75 &
sleep 10m
python launcher_gsac.py --env Walker2d-v3 --num_runs 5 -tsr 0.25 && python launcher_gsac.py --env Hopper-v3 --num_runs 5 -tsr 0.25 &
python launcher_gsac.py --env Walker2d-v3 --num_runs 5 -tsr 0.75 && python launcher_gsac.py --env Hopper-v3 --num_runs 5 -tsr 0.75 &


