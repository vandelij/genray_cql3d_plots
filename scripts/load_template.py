from ifigure.mto.hg_support import load_subtree_hg
import os
'''
load template input from repository and use
   # load template
   load_template(name, path= None, data_dest = None):
   # list available templates
   load_template(list = True)
'''
def list_template(path= None):
   if path is None:
       folder = obj.get_pyfolder()
       url, root, path, pathdir, pathbase = folder.get_hg_pathinfo()
   return os.listdir(os.path.join(root, pathdir, 'genray_cql3d_templates'))

def load_template(name, path = None, data_dest = None):
   if path is None:
       folder = obj.get_pyfolder()
       url, root, path, pathdir, pathbase = folder.get_hg_pathinfo()
   if data_dest is None:
       data_dest = model
   template = load_subtree_hg(model, root, 
                          os.path.join(pathdir, 'genray_cql3d_templates/'+name),
                          'template')

   model.scripts.copy_template(template)
   if model.has_child('data'): 
       model.data.destroy()
   if template.has_child('data'):
       template.data.move(data_dest)
   template.destroy()

do_list =kwargs.pop('list', None)
if do_list is None:
    ans(load_template(*args, **kwargs))
else:
    ans(list_template(*args, **kwargs))