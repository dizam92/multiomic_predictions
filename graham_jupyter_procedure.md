#  Go here if an error occur [https://github.com/tvhahn/compute-canada-hpc/blob/master/02-create-notebooks/02-notebook-setup.md]
##  Resume of the steps 

1. I already create the environment thus compute this: jup_env to activate the environment
2. Install whatever you want via pip install package-name
3. salloc command like this: salloc --time=03:00:00 --nodes=1  --ntasks-per-node=16 --mem=32G --account=rrg-corbeilj-ac srun jupyter_py3/bin/notebook.sh
4. Open another terminal and use this: ssh -L "port_number":"the grape connexion".graham.sharcnet:"port_number" maoss2@graham.computecanada.ca
5. Leave those 2 terminal OPEN all time (open another one if you wanna do things in your login)
6. Use the port available and the token like this in your browser: http://localhost:8888/?token=<token>
