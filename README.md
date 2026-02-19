Install as a package: 
`pip install -e [full path to personal_utils folder]`

Import:
`import overengineering`

If you want a specific module, such as general:
`from overengineering.general import *`

If the coloring isn't recognized in VS Code, add this to your workspace settings:
```
"python.analysis.extraPaths": [
			"[path to personal_utils folder]"
		]
```