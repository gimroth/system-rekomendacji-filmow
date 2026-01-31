import pkgutil, json
mods=[name for _,name,_ in pkgutil.iter_modules() if 'anfis' in name.lower() or 'xanfis' in name.lower()]
print(json.dumps(mods))
