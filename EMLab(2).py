"""
An integrated and interactive platform for electromagnetic simulation
An assignment for EM course
syuqiao22@qq.com
zhaoyifan@mail.ustc.edu.cn

Current available functions:
    *welcome*       | cr            | rm                | sm
    quit            |     -w        |   (number options)|   -p
    more            |     -s        |   -a              |       -i,c,e,p,m
    lscmd           |     -r        | umf               | si
    la              |               |   -c              |   -d,l
    *empty*         | uef           |   -i              | *delay*
    ex              |     -c        |                   | pl
    ls              |     -i        | *dtm*             |   -t
                    |               | *dte*             |   -f
                    |               |                   |       -e,m,d
Update 22     2024/12/16
Unsolved problem: two-body
"""

try:
#    import numpy as np
    import EMobjClasses
    import EMfuncs
    import time
except ImportError:
    print('Failed to import EMobjClasses.py or EMfuncs.py or time.py\n'+
          'Check if supplementary files is included in PATH variables')
    raise SystemExit(1)

print('Welcome to EM Lab!\n' + __doc__)
print('To find out how commands work, type "help"')

#funcslist = {
#    'bye': EMfuncs.quit,
#    'quit': EMfuncs.quit,
#    'ls' : EMfuncs.list_objs,
#    'list' : EMfuncs.list_objs,
#    'lscmd' :
#}
cmdlist = {}
count_cmds = 0
help_str = (
    "command 'help' : show this page or help document of other functions\n"
    "All commands:\n"
    "'help', '?' : show this page\n"
    "\tIf followed with command name, it will show the help document of the pertinent function\n"
    "'quit', 'bye' : quit EMlab\n"
    "'lscmd' : list executed command\n"
    "'la' : execute last command\n"
    "'ex' : execute all commands in the given txt file\n"
    "\tNote: the file cannot include command 'la' and 'ex'\n"
    "'ls' : list all objects with its object ID, type, and position\n"
    "'more' : show more attribute of an object\n"
    
)

def list_commands():
    print('CmdID | Content')
    for keys in cmdlist.keys():
        print('{0:>5} | {1}'.format(
            keys,
            cmdlist[keys]
        ))

def operate(user_command): #user_command: string
    global count_cmds
    count_cmds += 1
    cmdlist[count_cmds] = user_command

    if user_command in ('bye', 'quit'):
        try:
            EMfuncs.quit_lab()
        except EMfuncs.BreakLoop:
            raise EMfuncs.BreakLoop
    elif user_command == 'lscmd': list_commands()

    elif user_command.split(' ')[0] in ('ls', 'list'): EMfuncs.list_objs()
    elif user_command.split(' ')[0] == 'more':
        try:
            EMfuncs.show_more_attr_of_obj(int(user_command.split(' ')[1]))
        except Exception as e:
            match e:
                case ValueError():
                    print('Invalid argument')
                case KeyError():
                    print('Invalid argument')
                case _ : raise e

    elif user_command.split(' ')[0] in ('cr', 'create'):
        croption, uselessvar, flargs = EMfuncs.get_arguments(user_command)
        try:
            EMfuncs.create_obj(croption, flargs)
        except Exception as e:
            match e:
                case IndexError():
                    print('Invalid command')
                case ValueError():
                    print('Parameter error: parameters should be integers')
                case KeyError():
                    print('Option error: check whether option is valid')
                case TypeError() :
                    print('Type error: too many arguments')
                case EMobjClasses.FacingVectorZeroError() :
                    print('Facing vector cannot be zero')
                case _ :
                    raise e
    elif user_command.split(' ')[0] == 'uef':
        cruefoption, uselessvar, flargs = EMfuncs.get_arguments(user_command)
        try:
            EMfuncs.create_UEF(cruefoption, flargs)
        except Exception as e:
            match e:
                case IndexError():
                    print('Invalid command')
                case ValueError():
                    print('Parameter error: parameters should be integers')
                case KeyError():
                    print('Option error: check whether option is valid')
                case AssertionError():
                    print('Coordinates of border must greater than that of base')
                case EMobjClasses.FacingVectorZeroError():
                    print('Facing vector cannot be zero')
                case _ :
                    raise e
    elif user_command.split(' ')[0] == 'umf':
        crumfoption, uselessvar, flargs = EMfuncs.get_arguments(user_command)
        try:
            EMfuncs.create_UMF(crumfoption, flargs)
        except Exception as e:
            match e:
                case IndexError():
                    print('Invalid command')
                case ValueError():
                    print('Parameter error: parameters should be integers')
                case KeyError():
                    print('Option error: check whether option is valid')
                case AssertionError():
                    print('Coordinates of border must greater than that of base')
                case EMobjClasses.FacingVectorZeroError():
                    print('Facing vector cannot be zero')
                case _:
                    raise e
    elif user_command.split(' ')[0] in ('sm', 'summon'):
        smoption, has_oo, *ptt_oo_and_flargs= EMfuncs.get_arguments(user_command, do_return_optional_option=True)
        try:
            EMfuncs.create_dynamic_obj(smoption, ptt_oo_and_flargs, has_oo)
        except Exception as e:
            match e:
                case IndexError(): print('Invalid command: index error')
                case ValueError(): print('Parameter error: parameters should be integers')
                case KeyError(): print('Option error: check whether option is valid')
                case TypeError(): print('Type error: too many arguments')
                case _ :
                    raise e

    #This function doesn't adopt get_arguments() intentionally
    elif user_command.split(' ')[0] in ('rm', 'del'):
        try:
            if user_command.split(' ')[1] == '-a':
                EMfuncs.remove_all_obj()
            else:
                try:
                    EMfuncs.remove_obj(user_command.split(' ')[1:])
                except Exception as e:
                    match e:
                        case KeyError():
                            print('Nonexistent object')
                        case ValueError():
                            print('Parameter error: parameters should be integers')
                        case _:
                            raise e
        except IndexError:
            print('Invalid command')

    elif user_command.split(' ')[0] == 'dte':
        location = [float(i) for i in user_command.split(' ')[1:4]]
        print('Electric field intensity at {0}: {1}'.format(location, EMobjClasses.total_measure_Efield(location)))
    elif user_command.split(' ')[0] == 'dtm':
        location = [float(i) for i in user_command.split(' ')[1:4]]
        print('Magnetic field intensity at {0}: {1}'.format(location, EMobjClasses.total_measure_Mfield(location)))

    elif user_command.split(' ')[0] in ('simu', 'si'):
        simuoption, uselessvar, flargs = EMfuncs.get_arguments(user_command)
        try:
            EMfuncs.simulate(simuoption, flargs)
        except Exception as e:
            match e:
                case KeyError(): print('Invalid option')
                case TypeError(): print('Type error: invalid argument')
                case IndexError(): print('Invalid argument: objID does not exist')
                case _ : raise e

    elif user_command.split(' ')[0] in ('dela', 'delay'): time.sleep(float(user_command.split(' ')[1]))

    elif user_command.split(' ')[0] in ('pl', 'plot') :
        try:
            plot_option, has_op_option, *ptt_oo_and_flargs = EMfuncs.get_arguments(user_command,
                                                                                   do_return_optional_option=True,
                                                                                   do_return_string_args=True)
            # print(plot_option, ptt_oo_and_flargs, has_op_option)
        except Exception as e:
            match e:
                case IndexError():
                    print('Invalid command: index error')
                case ValueError():
                    print('Parameter error: parameters should be integers')
                case KeyError(): print('Key error: objID does not exist')
                case _ :
                    raise e
            return 114514
        try:
            EMfuncs.plot_main(plot_option, ptt_oo_and_flargs, has_optional_option=has_op_option)
        except Exception as e:
            match e:
                case IndexError(): print('Invalid argument')
                case TypeError(): print('Type error: invalid argument')
                case ValueError(): print('Parameter error: parameters should be integers '
                                         'or floats')
                case NameError(): print('Invalid command: lack of crucial arguments')
                case _ : raise e

    elif user_command.split(' ')[0] in ('help', '?'):
        if len(user_command.split(' ')) == 1:
            print(help_str)
    else:
        if not user_command == 'empty':
            print('Unknown command')
        del cmdlist[count_cmds]
        count_cmds -= 1


while True:
    command = input('> ').strip()
    if command == 'la':
        try:
            command = cmdlist[count_cmds]
        except KeyError:
            print('Invalid command: first command cannot be "la"')
    elif command.split(' ')[0] in ('ex', 'exec', 'execute'):
        execfilename = command.strip().split(' ')[1].replace('"', '')
        print('executing: ' + execfilename + '\n')
        try:
            with open(execfilename) as f:
                for (linenumber, line)  in enumerate(f.readlines(), start=1):
                    print('Line ' + str(linenumber) + ': ', end='')
                    operate(line.strip())
                    print('')
        except Exception as e2:
            match e2:
                case FileNotFoundError():
                    print('File not found: ' + execfilename)
                case EMfuncs.BreakLoop():
                    break
                case _ :
                    raise e2
        finally:
            command = 'empty'

    try:
        operate(command)        #cmd are counted into dict here
    except Exception as e2:
        match e2:
            case EMfuncs.BreakLoop():
                break
            case _ :
                raise e2


