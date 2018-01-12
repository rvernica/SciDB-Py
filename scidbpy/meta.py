# List of hungry operators, they execute immediately, e.g.,
# ``db.remove``.
#
# Operators **not** listed here are lazy. They will execute at a later
# time.
#
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

# List of operators with string arguments. The list is groupped by
# argument position. If an operator has multiple string arguments, it
# is listed multiple times. If an argument can be a string or somting
# else, it is excluded from this list.
#
string_args = (
    # 1st argument
    set((
        # list('operators');
        # ---
        'cancel',
        'list',
        'load_library',
        'load_module',
        'unload_library',

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
        'set_namespace',
        'set_role',
        'set_role_permissions',
        'show_users_in_role',
    )),
    # 2nd argument
    set((
        # list('operators');
        # ---
        'input',
        'load',
        'save',
        'show',

        # load_library('namespaces');
        # ---
        'add_user_to_role',
        'change_user',
        'create_user',
        'drop_user_from_role',
        'move_array_to_namespace',
        'set_role_permissions',
    )),
    # 3rd argument
    set((
        # load_library('namespaces');
        # ---
        'change_user',
        'set_role_permissions',
    )),
    # 4thd argument
    set((
        # list('operators');
        # ---
        'input',
        'load',
        'save',

        # load_library('namespaces');
        # ---
        'set_role_permissions',
    )),
    # 5th argument
    set((
    )),
    # 6th argument
    set((
    )),
    # 7th argument
    set((
    )),
    # 8th argument
    set((
    ))
)
