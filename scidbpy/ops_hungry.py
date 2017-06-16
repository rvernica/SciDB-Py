# list of hungry operators, they execute immediately, e.g.,
# "db.remove(...)", they are not lazy
ops_hungry = (
    # list('operators');
    # ---
    'cancel',

    'consume',
    'create_array',
    'create_array_using',

    'delete',

    'help',

    'insert',

    'load_library',
    'load_module',

    'remove',
    'remove_versions',
    'rename',

    'save',

    'store',

    'unload_library',

    # list('macros');
    # ---
    'load',

    # load_library('dense_linear_algebra');
    # ---
    'mpi_init',

    # load_library('namespaces');
    # ---
    'add_user_to_role',
    'change_user',
    'create_namespace',
    'create_role',
    'create_user',
    'drop_namespace',
    'drop_role',
    'drop_user',
    'drop_user_from_role',
    'move_array_to_namespace',
    'set_namespace',
    'set_role_permissions',
    )
