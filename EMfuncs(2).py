"""funcs for EMlab project"""

import EMobjClasses as emc
import datetime
import sys
import matplotlib.pyplot as plt
import numpy as np
import os
from mpl_toolkits.mplot3d import Axes3D

class BreakLoop(Exception): pass
def quit_lab():
    print('Thank you for using EM Lab!')
    raise BreakLoop

title_display_string = 'ObjID | ' + ' '*8 +'Type' + ' '*8 + ' | ' + ' '*5 +'location'
def list_objs():
    print(title_display_string)
    for keys in emc.dict_existed_objs.keys():
        print('{0:>5} | {1:>20} | {2}'.format(
            keys,
            emc.dict_existed_objs[keys].__class__.__name__,
            emc.dict_existed_objs[keys].position
        ))
def show_more_attr_of_obj(objID):
    target_obj = emc.dict_existed_objs[objID]
    print('Show more attributes of objID: {0}'.format(objID))
    for attr in target_obj.__dict__:
        if not attr.startswith('_'):
            print('\t{0}: {1}'.format(attr, target_obj.__dict__[attr]))

def macro_converter(token):
    find_pounde = token.find('#e')
    if find_pounde == 0:
        res = str(emc.proton_charge)
    elif find_pounde == -1:
        res = token
    else:
        res = str(float(token[:find_pounde]) * emc.proton_charge)
    return res
def get_arguments(command_string,
                  do_return_optional_option=False,
                  do_return_string_args=False):
    #Command string: include title 'cr' etc.
    tokens_list = command_string.strip().split(' ')
    has_optional_option = tokens_list[2].startswith('-') and not tokens_list[2][1].isdigit()
    #Macro substitution
    res_tokens_list = [macro_converter(token) for token in tokens_list]

    # print(res_tokens_list)

    if has_optional_option and do_return_optional_option:
        if do_return_string_args:
            return (res_tokens_list[1][1:],
                True,
                [float(token) for token in res_tokens_list[3:] if not (token.startswith('::') or token.startswith('"'))],
                res_tokens_list[2][1:],
                [token for token in res_tokens_list[3:] if token.startswith('"')]
                    )
        else:
            return (res_tokens_list[1][1:],
                    True,
                    [float(token) for token in res_tokens_list[3:] if not (token.startswith('::'))],
                    res_tokens_list[2][1:])
    else:
        if do_return_optional_option:
            return (res_tokens_list[1][1:],
                    False,
                    [float(token) for token in res_tokens_list[2:] if not (token.startswith('::') or token.startswith('"'))],
                    [token for token in res_tokens_list[2:] if token.startswith('"')]
                    )
        else:
            return (res_tokens_list[1][1:],
                    False,
                    [float(token) for token in res_tokens_list[2:] if not token.startswith('::')],
                    )
# def list_commands():
#     print('CmdID | Content')
#     for keys in cmdlist.keys():
#         print('{0:>5} | {1}'.format(
#             keys,
#             cmdlist[keys]
#         ))

# def total_measure_Efield(loca):
#     res = (0, 0, 0)
#     for objid in emc.dict_existed_objs.keys():
#         component = emc.dict_existed_objs[objid].indiv_measure_Efield(loca)
#         res = tuple([i + j for i, j in zip(res, component)])
#     return res
# def total_measure_Mfield(loca):
#     res = (0, 0, 0)
#     for objid in emc.dict_existed_objs.keys():
#         component = emc.dict_existed_objs[objid].indiv_measure_Mfield(loca)
#         res = tuple([i + j for i, j in zip(res, component)])
#     return res

def remove_obj(targets):
    """

    :param targets:['int', 'int', ...]
    :return:
    """

    for keys in targets:
        keys = int(keys)
        print('removing obj: %d' % keys)
#        print(emc.dict_existed_objs)#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!debug code
        del emc.dict_existed_objs[keys]
#    total_update_Efield()
#    total_update_Mfield()
def remove_all_obj():
    alltargets = [keys for keys in emc.dict_existed_objs.keys()]
    remove_obj(alltargets)

def create_obj(objtype, flargs):
    """
    command 'cr': create static objects
    :input cr -(object type) [location x] [location y] [location z] {type-specific args}
    :param objtype: 'w'-> Charged/Galvanized Wire, '-s'->Charged Sphere
    :param flargs:
    type-specific args list:
        'w': [facing vector x] [facing vector y] [facing vector z] [linear charge density] [current]
                if current direction reverses against facing direction, input a negative current
        's': [facing vector x] [facing vector y] [facing vector z] [radius] [total charge]
        'r': [facing vector x] [facing vector y] [facing vector z] [radius] [linear charge density] [current]

    :return:
    """
    type_match_list = {
        'w' : emc.ChargedWire,
        's' : emc.ChargedConductSphere,
        'r' : emc.GalvanizedRing
    }

    print('creating object: type-{0}'.format(type_match_list[objtype[0]].__name__))
    emc.dict_existed_objs[emc.count_existed_objs] = type_match_list[objtype[0]](*flargs)

    print('created object:\n' +
            title_display_string + ' '*16 + '\n' +
            str(emc.dict_existed_objs[emc.count_existed_objs]))
def create_dynamic_obj(objtype, flargs_and_pttoo, has_optional_option=False):
    """
    command 'sm': create dynamic objects
    sm -p -[options] [location x] [location y] [location z] [velocity x] [velocity y] [velocity z] [acceleration x] [acceleration y] [acceleration z] [charge] [mass]
        (1 compulsory option, 0 ~ 5 optional options and 5/11 parameters at all)
        Optional options:
            'c': you needn't input 3 velocity parameters and 3 acceleration parameters, they are set as 0
            'i': during simulation the electric and magnetic field of particle will be updated real-time
            'e': set the unit of charge as proton charge
            'm': set the unit of mass as electron mass
            'p': set the unit of mass as proton mass. This will mask optional option 'm'
    :param objtype: 'p' : charged particle
    :param flargs_and_pttoo:
    :param has_optional_option:
    :return:
    """
    if objtype[0] == 'p':
        optional_options = flargs_and_pttoo[1] if has_optional_option else ''
        flargs = flargs_and_pttoo[0]
        print('creating moving object: type-Dynamic particle')

        lenflargs = len(flargs)
        argsamount = 5 if ('c' in optional_options) else 11

        if lenflargs <= argsamount - 1:
            emc.dict_existed_objs[emc.count_existed_objs] = emc.DynamicParticle(*flargs)
            # flargs = (flargs + list(emc.posi_default)[lenflargs - 3] + [0])
        else:
            emc.dict_existed_objs[emc.count_existed_objs] = emc.DynamicParticle(*(flargs[:argsamount - 2]),
                        charge = flargs[argsamount - 2] * (emc.proton_charge if ('e' in optional_options) else 1),
                        mass = flargs[argsamount - 1] * (emc.proton_mass if ('p' in optional_options) else (
                            emc.electron_mass if ('m' in optional_options) else 1
                        )),
                        isCountEField = ('i' in optional_options))
        print('created dynamic particle:\n' + title_display_string + ' ' * 16 + '\n' +
              str(emc.dict_existed_objs[emc.count_existed_objs]))
    else:
        raise KeyError

def create_UEF(field_shape, flargs):
    """
    command 'uef': create UEF objects
    uef -c [base x] [base y] [base z] [border x] [border y] [border z] [field vector x] [field vector y] [field vector z] [intensity]
        (1 option and 10 args at all)
        Any coordinate of border should be greater than base coordinate
        '::' for annotation
    uef -i [field vector x] [field vector y] [field vector z] [intensity]
        (1 option and 4 args at all)
    :param field_shape: '-c'-> Cubic field, '-i'-> Infinite field,
    :param flargs:
    :return:
    """
    if field_shape[0] == 'c':
        assert (flargs[3] > flargs[0]
            and flargs[4] > flargs[1]
            and flargs[5] > flargs[2]), "Coordinates of border must greater than that of base"
        if (flargs[6], flargs[7], flargs[8]) == (0, 0, 0):
            raise emc.FacingVectorZeroError
        print('creating uniform electric field: shape-Cubic')
        emc.dict_existed_objs[emc.count_existed_objs] = emc.CubicUniEField(*flargs)
        print('created uniform electric field:\n' + title_display_string + ' ' * 16 + '\n' +
              str(emc.dict_existed_objs[emc.count_existed_objs]))

    elif field_shape[0] == 'i':
        flargs = [0, 0, 0] + flargs
        if (flargs[3], flargs[4], flargs[5]) == (0, 0, 0):
            raise emc.FacingVectorZeroError
        print('creating uniform electric field: shape-Infinite')
        emc.dict_existed_objs[emc.count_existed_objs] = emc.InfiniteUniEField(*flargs)
        print('created uniform electric field:\n' + title_display_string + ' ' * 16 + '\n' +
              str(emc.dict_existed_objs[emc.count_existed_objs]))
    else:
        raise KeyError
def create_UMF(field_shape, flargs):
    """
    command 'umf': create UMF objects
    umf -c [base x] [base y] [base z] [border x] [border y] [border z] [field vector x] [field vector y] [field vector z] [intensity]
        (1 option and 10 args at all)
        Any coordinate of border should be greater than base coordinate
    umf -i [field vector x] [field vector y] [field vector z] [intensity]
        (1 option and 4 args at all)
    :param field_shape: '-c'-> Cubic field, '-i'-> Infinite field,
    :param flargs:
    :return:
    """
    if field_shape[0] == 'c':
        assert (flargs[3] > flargs[0]
            and flargs[4] > flargs[1]
            and flargs[5] > flargs[2]), "Coordinates of border must greater than that of base"
        if (flargs[6], flargs[7], flargs[8]) == (0, 0, 0):
            raise emc.FacingVectorZeroError
        print('creating uniform magnetic field: shape-Cubic')
        emc.dict_existed_objs[emc.count_existed_objs] = emc.CubicUniMField(*flargs)
        print('created uniform magnetic field:\n' + title_display_string + ' ' * 16 + '\n' +
              str(emc.dict_existed_objs[emc.count_existed_objs]))

    elif field_shape[0] == 'i':
        flargs = [0, 0, 0] + flargs
        if (flargs[3], flargs[4], flargs[5]) == (0, 0, 0):
            raise emc.FacingVectorZeroError
        print('creating uniform magnetic field: shape-Infinite')
        emc.dict_existed_objs[emc.count_existed_objs] = emc.InfiniteUniMField(*flargs)
        print('created uniform magnetic field:\n' + title_display_string + ' ' * 16 + '\n' +
              str(emc.dict_existed_objs[emc.count_existed_objs]))
    else:
        raise KeyError

steplength_default = 0.0625
endtime_default = 8


def default_print_simuresult(simutime, statuses, output_file, is_first_line=True):
    """
    put simu results into the default file

    # in simu_result_20241206_221912.dat
    1 | 2 | 3
    0.0, 1.0, 0.0, 0.0 | 0.0,2.0,0.0,0.0 | 0.0, 3.0, 0.0, 0.0
    0.0625, 1.0, 1.0, 0.0 | 0.0625, 2.0, 1.0, 0.0 | 0.0625, 3.0, 1.0, 0.0
    ...

    :param simutime:
    :param statuses: a dict: {objID: ((accele), (velo), (position)), ...}
    :param output_file:
    :param is_first_line: if True-> the first line of file
                          if False -> the rest lines of file
    :return:
    """
    if is_first_line:
        output_file.write(b'Simulated ObjID:')
        for objID in statuses.keys():
            output_file.write(str(objID).encode())
            output_file.write(b' | ')
        output_file.write(b'\n')
    for objID in statuses.keys():
        output_file.write(str(simutime).encode())
        output_file.write(b', ')
        output_file.write(str(statuses[objID][-1])[1:-1].encode())
        output_file.write(b' | ')
    output_file.write(b'\n')
    output_file.flush()
def judge_if_change_steplength(emobj, steplength): return False
def change_steplength(steplength, *args): return steplength
def update_statuses_of_objs(dyna_objects, steplength):
    builtin_aftermoving_statuses = {}
    do_change_steplength = False
    #Calculate differences
    for objID in dyna_objects:
        objID = int(objID)
        builtin_aftermoving_statuses[objID] = emc.dict_existed_objs[objID].calculate_next_dyna_status(
            steplength,
            do_update=False
        )
        if (builtin_aftermoving_statuses[objID][1][0] > 3e8 or
            builtin_aftermoving_statuses[objID][1][1] > 3e8 or
            builtin_aftermoving_statuses[objID][1][2] > 3e8):
            print('\nToo fast')
            raise BreakLoop()
    # Update statuses
    for objID in dyna_objects:
        emc.dict_existed_objs[objID].accl = builtin_aftermoving_statuses[objID][0]
        emc.dict_existed_objs[objID].velo = builtin_aftermoving_statuses[objID][1]
        emc.dict_existed_objs[objID].position = builtin_aftermoving_statuses[objID][2]
        if judge_if_change_steplength(emc.dict_existed_objs[objID], steplength):
            do_change_steplength = True

    return builtin_aftermoving_statuses, do_change_steplength

def simulate(simu_option, flargs):
    """
    command 'si': simulate the movement of particles
    si -[simu options] [end time] [step length] [object ID 1] [object ID 2] ...

    :param flargs: a list of floats
    :param simu_option:
    Incompatible:
        'l' -> './simu_result_%Y%m%d_%H%M%S.dat', with literal output;
    Compatible:
        'd' -> use default end time and step length;
    :return:

    WARNING: if you simulate multiple tasks within 1 ms, only the last one will be preserved
    """
    print('Simulating objects...')
    #variables declamation
    now = datetime.datetime.now()
    standard_output_file_name = now.strftime('./simu_result_%Y%m%d_%H%M%S_%f.dat')
    simutime = 0
    do_change_steplength = False
    output_file_name = standard_output_file_name
    if 'd' in simu_option:
        end_time = endtime_default
        steplength = steplength_default
        dyna_objects = [int(arg) for arg in flargs]
    else:
        end_time = flargs[0]
        steplength = flargs[1]
        dyna_objects = [int(arg) for arg in flargs[2:]]

    aftermoving_statuses = {objID: (emc.dict_existed_objs[objID].accl,
                                    emc.dict_existed_objs[objID].velo,
                                    emc.dict_existed_objs[objID].position)
                            for objID in dyna_objects}

    if 'l' in simu_option:
        printer = default_print_simuresult
    else:
        raise KeyError

    with open(output_file_name, 'wb') as output_file:
        printer(simutime, aftermoving_statuses, output_file)
        while simutime < end_time:
            try:
                (aftermoving_statuses, do_change_steplength) = update_statuses_of_objs(dyna_objects, steplength)
            except BreakLoop:
                break
            #Print results and progression bar
            simutime += steplength
            printer(simutime, aftermoving_statuses, output_file, is_first_line=False)
            sys.stdout.write('\r'+'Simulation time progress: '+str(simutime)+'/'+str(end_time))
            sys.stdout.flush()

            # Change steplength
            if do_change_steplength:
                steplength = change_steplength(steplength, [])
    print('\nDone!')

default_plot_range = (-1,-1,-1,1,1,1)
default_plot_arguments = {
    'pl_per_axis' : 10,
    'arrow_length' : 0.1,
    'line_width' : 0.7
}
def plot_EMfield(field_type, pl_per_axis=default_plot_arguments['pl_per_axis'],
                 arrow_length=default_plot_arguments['arrow_length'],
                 line_width=default_plot_arguments['line_width'],
                 plot_range=default_plot_range,
                 figure_name=None):
    """

    :param plot_range: (x_inf, y_inf, z_inf, x_sup, y_sup, z_sup)
    :param pl_per_axis:
    :param arrow_length:
    :param line_width:
    :param field_type:
    :param figure_name:
    :return:
    """
    now=datetime.datetime.now()

    match field_type:
        case 'e':
            func_measure_field_type = emc.total_measure_Efield
            if not figure_name:
                save_file_name = now.strftime('plot_electro_field_%Y%m%d_%H%M%S_%f.eps')
            else:
                save_file_name = figure_name
        case 'm':
            func_measure_field_type = emc.total_measure_Mfield
            if not figure_name:
                save_file_name = now.strftime('plot_magnet_field_%Y%m%d_%H%M%S_%f.eps')
            else:
                save_file_name = figure_name
        case _ :
            raise AssertionError

    ax = plt.figure().add_subplot(projection='3d')

    # Make data
    total_plots_per_axis = pl_per_axis
    x = np.linspace(plot_range[0], plot_range[3], total_plots_per_axis)
    y = np.linspace(plot_range[1], plot_range[4], total_plots_per_axis)
    z = np.linspace(plot_range[2], plot_range[5], total_plots_per_axis)
    plotted_indvar_X, plotted_indvar_Y, plotted_indvar_Z = np.meshgrid(x, y, z)

    cut_threshold = 50

    U, V, W = func_measure_field_type((plotted_indvar_X, plotted_indvar_Y, plotted_indvar_Z), cut_threshold=cut_threshold)

    # Plot
    ax.quiver(plotted_indvar_X, plotted_indvar_Y, plotted_indvar_Z, U, V, W,
              length=arrow_length,
              linewidth=line_width)

    ax.set(xticklabels=[],
           yticklabels=[],
           zticklabels=[])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(save_file_name.split('.')[0])
    try:
        plt.savefig('./figs/' + save_file_name, dpi=300)
    except FileNotFoundError:
        os.makedirs('./figs')
        plt.savefig('./figs/' + save_file_name, dpi=300)
    plt.show()

def plot_particle_trajectory(simulated_particles_list,
                             endtime=8,
                             steplength=0.0625,
                             plot_range=default_plot_range,
                             figure_name=None):
    """
    command 'pl': plot trajectory of given particles / field of all objects
    pl -t -[optional options] [range x inferior] [y inf] [z inf] [x superior] [y sup] [z sup] [end time] [steplength] [particle 1] [particle 2] [...]

    :param simulated_particles_list:
    :param endtime:
    :param steplength:
    :param plot_range:
    :param figure_name:
    :return:
    """

    # simulated_particles_list = [int(arg) for arg in simulated_particles_list]

    amount_simulated_particles = len(simulated_particles_list)
    amount_to_obj_map = {i : emc.dict_existed_objs[simulated_particles_list[i]]
                         for i in range(amount_simulated_particles)}
    num_steps = int(endtime / steplength)

    # print(amount_to_obj_map)

    positions = np.zeros((num_steps, amount_simulated_particles, 3))
    # velocities = np.zeros((num_steps, amount_simulated_particles, 3))
    buffers = np.zeros((amount_simulated_particles, 3, 3))
    # buffers:
    # [*objects to be simulated
    #       [[], #accl
    #        [], #velo
    #        []],#position
    for i in range(amount_simulated_particles):
        positions[0][i] = np.array(amount_to_obj_map[i].position)
        # velocities[0][i] = np.array(amount_to_obj_map[i].velo)

    # 计算运动轨迹
    for step in range(1, num_steps):
        for i in range(amount_simulated_particles):
            buffers[i] = np.array(amount_to_obj_map[i].calculate_next_dyna_status(steplength, do_update=False))
            positions[step][i] = np.array(buffers[i][2])
            # print(buffers[i][1])

        for i in range(amount_simulated_particles):
            amount_to_obj_map[i].accl = tuple(buffers[i][0])
            amount_to_obj_map[i].velo = tuple(buffers[i][1])
            amount_to_obj_map[i].position = tuple(buffers[i][2])
        print('\r' + str(step) + ' / ' + str(num_steps), end='')

    if figure_name:
        save_figure_name = figure_name
    else:
        now = datetime.datetime.now()
        save_figure_name = now.strftime('plot_particle_trajectory_%Y%m%d_%H%M%S.eps')
    # print(positions)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(amount_simulated_particles):
        ax.plot(positions[:, i, 0], positions[:, i, 1], positions[:, i, 2])
    ax.set_xlim(xmin=plot_range[0], xmax=plot_range[3])
    ax.set_ylim(ymin=plot_range[1], ymax=plot_range[4])
    ax.set_zlim(zmin=plot_range[2], zmax=plot_range[5])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title(save_figure_name.split('.')[0])
    try:
        plt.savefig('./figs/' + save_figure_name, dpi=300)
    except FileNotFoundError:
        os.makedirs('./figs')
        plt.savefig('./figs/' + save_figure_name, dpi=300)
    plt.show()
def plot_main(plot_option, flargs_and_pttoo, has_optional_option=False):
    """
    command 'pl' : plot trajectory of given particles / field of all objects
    pl -t -[optional options] [range x inferior] [y inf] [z inf] [x superior] [y sup] [z sup] [end time] [steplength] [particle 1] [particle 2] [...] [file name]
        Optional options:
            'd' : use default range of plot
    pl -f -[optional options] [range x inferior] [y inf] [z inf] [x superior] [y sup] [z sup] [plot per axis] [arrow length] [line width] [file name]
        Optional options:
            'e' : plot total electronic field of all objects
            'm' : plot total magnetic field of all objects
            'd' : use default range of plot
    :param flargs_and_pttoo:
    [[float, float, ],
    'optional option', #potential
    [string, string]
    ]
    file name should be contained by ""
    :param has_optional_option:
    :param plot_option:
    """
    is_any_plot_executed = False
    if 't' in plot_option:

        optional_option = flargs_and_pttoo[1] if has_optional_option else ''
        flargs = flargs_and_pttoo[0]
        strargs = flargs_and_pttoo[2] if has_optional_option else flargs_and_pttoo[1]

        cons_argu_amount = 2 if 'd' in optional_option else 8
        list_of_particle_plotted = [int(i) for i in flargs[cons_argu_amount : ]]
        print('plotting particle trajectory of ' + str(list_of_particle_plotted))

        plot_particle_trajectory(list_of_particle_plotted,
                                 endtime=flargs[cons_argu_amount - 2],
                                 steplength=flargs[cons_argu_amount - 1],
                                 plot_range=flargs[ : 6] if not 'd' in optional_option else default_plot_range,
                                 figure_name=strargs[0][1:-1] if strargs else None
                                 )
        is_any_plot_executed = True
    if 'f' in plot_option:
        optional_option = flargs_and_pttoo[1] if has_optional_option else ''
        flargs = flargs_and_pttoo[0]
        strargs = flargs_and_pttoo[2] if has_optional_option else flargs_and_pttoo[1]

        const_argu_amount = 3 if 'd' in optional_option else 9
        if 'e' in optional_option:
            print('plotting electronic field ...')
            plot_EMfield('e',
                         pl_per_axis=int(flargs[const_argu_amount - 3]),
                         arrow_length=flargs[const_argu_amount - 2],
                         line_width=flargs[const_argu_amount - 1],
                         plot_range=flargs[ : 6] if not 'd' in optional_option else default_plot_range,
                         figure_name=strargs[0][1:-1] if strargs else None
                         )
            is_any_plot_executed = True
        if 'm' in optional_option:
            print('plotting magnetic field ...')
            plot_EMfield('m',
                         pl_per_axis=int(flargs[const_argu_amount - 3]),
                         arrow_length=flargs[const_argu_amount - 2],
                         line_width=flargs[const_argu_amount - 1],
                         plot_range=flargs[ : 6] if not 'd' in optional_option else default_plot_range,
                         figure_name=strargs[0][1:-1] if strargs else None
                         )
            is_any_plot_executed = True
    if not is_any_plot_executed:
        print('Nothing executed. Check if options are valid')


if __name__ == '__main__':
    create_obj('r', [0, 0, -2, 0, 0, 1, 1, 1, 5e6])
    # print(emc.total_measure_Mfield((0, 0, 0)))
    # create_obj('r', [0, 0, 2, 0, 0, 1, 1, 1, -5e6])
    # print(emc.total_measure_Mfield((0, 0, 0)))
    # create_UEF('i', [0, 0, 1, 0.2])
    create_dynamic_obj('p', [[0, 0, 0, 1, 0, 0, 0, 0, 0, 1e1, 1]])
    create_dynamic_obj('p', [[0, 0, 0, -1, 0, 0, 0, 0, 0, 1e1, 1]])
    # plotEMfield('m', plot_range=(-2, -2, -3, 2, 2, 3), pl_per_axis=13)
    plot_EMfield('m')
    plot_particle_trajectory([2, 3], 8, 1, (-1, -1, -1, 1, 1, 1))