"""Classes and global vars for EMLab project
   Structure:
   func total_measure_Efield
   func total_measure_Mfield
   EMobj--
            EMstatic--
                    ChargedWire
                    UniEField
                        CubicUniEField
                        InfiniteUniEField
                    UniMField
                        CubicUniMField
                        InfiniteUniMField
                    GalvanizedRing
                    ChargedConductSphere
            EMdynamic--
                    DynamicParticle

            """
import math
import numpy as np
from PIL.ImageChops import difference

permit_of_vacuum = 8.8541878e-12
permea_of_vaccum = 4 * math.pi * 1e-7
proton_charge = 1.60218e-19
proton_mass = 1.67262e-27
electron_mass = 9.10596e-31

#global count_existed_objs
count_existed_objs = 0
dict_existed_objs = {}

posi_default = (0, 0, 0)
class EMobj:
    def __init__(self, position=posi_default):
        global count_existed_objs
        self.objID = count_existed_objs + 1
        count_existed_objs += 1
        dict_existed_objs[self.objID] = self
        self.position = position
    def __str__(self):
        return '{0:>5} | {1:>17} | {2} '.format(
            self.objID,
            self.__class__.__name__,
            self.position
        )
    def __del__(self):
        tempstr = str(self)
#        tempobjID = self.objID
#        print(dict_existed_objs)#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!ddebug code
#        del dict_existed_objs[tempobjID]
        print('removed object:' + tempstr)
        global count_existed_objs
        count_existed_objs -= 1

facing_default = (0, 0, 1)
class EMstatic(EMobj):
    def __init__(self, position=posi_default, facing=facing_default):
        EMobj.__init__(self, position)
        self.facing = facing

velo_default = (0, 0, 0)
accl_default = (0, 0, 0)

def total_measure_Efield(loca, cut_threshold=50):
    res = (0, 0, 0)
    for objid in dict_existed_objs.keys():
        component = dict_existed_objs[objid].indiv_measure_Efield(loca, cut_threshold=cut_threshold)
        # print('debug_tme: ' + str(res) + ' ' + str(component))
        res = tuple([i + j for i, j in zip(res, component)])
    return res
def total_measure_Mfield(loca, cut_threshold=50):
    res = (0, 0, 0)
    for objid in dict_existed_objs.keys():
        component = dict_existed_objs[objid].indiv_measure_Mfield(loca, cut_threshold=cut_threshold)
        res = tuple([i + j for i, j in zip(res, component)])
    return res

def cross_product(vec1, vec2, scale=1.0):
    return (
        (vec1[1]*vec2[2] - vec1[2]*vec2[1]) * scale,
        (vec1[2]*vec2[0] - vec1[0]*vec2[2]) * scale,
        (vec1[0]*vec2[1] - vec1[1]*vec2[0]) * scale,
    )
def scale_product(vec, scale):
    return (vec[0] * scale, vec[1] * scale, vec[2] * scale)
def vector_add(*vecs, dimension=3):
    result_vec = [0 for _ in range(dimension)]
    for i in range(dimension):
        for vec in vecs:
            result_vec[i] += vec[i]
    return tuple(result_vec)
def vector_substract(vec1, vec2):
    return tuple([i - j for i, j in zip(vec1, vec2)])
def dot_product(vec1, vec2):
    sum = 0
    for i,j in zip(vec1, vec2):
        sum += i * j
    # print('debug .* :' + str(sum))
    return sum
def measure_loca_difference(loca1, loca2):
    return tuple([j - i for i, j in zip(loca1, loca2)])
def mixed_product(vec1, vec2, vec3):
    return dot_product(vec1, cross_product(vec2, vec3))
#class LinearRelated(Exception): pass
def schmitt(vec1, vec2, vec3):
    univec1 = scale_product(vec1, 1 / np.sqrt(dot_product(vec1, vec1)))
    tempvec2 = vector_substract(vec2, scale_product(univec1, dot_product(vec2, univec1)))
    univec2 = scale_product(tempvec2, 1 / np.sqrt(dot_product(tempvec2, tempvec2)))
    tempvec3 = vector_substract(
        vec3, vector_add(scale_product(univec1, dot_product(vec3, univec1)),
                               scale_product(univec2, dot_product(vec3, univec2))
                               )
    )
    univec3 = scale_product(tempvec3, 1 / np.sqrt(dot_product(tempvec3, tempvec3)))
    return univec1, univec2, univec3
def single_schmitt(vec1):
    if mixed_product(vec1, (1, 0, 0), (0, 1, 0)) != 0:
        axisvec2, axisvec3 = (1, 0, 0), (0, 1, 0)
    elif mixed_product(vec1, (0, 1, 0), (0, 0, 1)) != 0:
        axisvec2, axisvec3 = (0, 1, 0), (0, 0, 1)
    else:
        axisvec2, axisvec3 = (0, 0, 1), (1, 0, 0)
    return schmitt(vec1, axisvec2, axisvec3)

def line_vector_integrate(func, var_start, var_end=2*math.pi, var_pieces=3600, do_show_process=False, **kargs):
    """

    :param func: func(variable for integration, **kargs) -> ( , , )
    :param var_start:
    :param var_end:
    :param var_pieces:
    :param kargs:
    :param do_show_process
    :return:
    """
    integrated_var = var_start
    var_steplength = (var_end - var_start) / var_pieces
    res = (0, 0, 0)
    while integrated_var < var_end:
        res = vector_add(res,
                         scale_product(func(integrated_var, **kargs), var_steplength)
                         )
        if do_show_process:
            print('\r' + 'integrating: ' + str(integrated_var) + ' / ' + str(var_end), end='' )
        integrated_var += var_steplength
    if do_show_process: print('')
    return res
def integrated_func(integrated_var, obj_instance, **kwargs):
    """

    :param obj_instance:
    have method 'func_provide_diffrnt_plot' and 'func_provide_plot_on_obj'
    THe first position of these methods should be the variable for integration
    :param kwargs: contain 'loca'
    :param integrated_var:
    :return:
    """

    loca = kwargs['loca']

    difference_arc = obj_instance.func_provide_diffrnt_plot(integrated_var, **kwargs)
    delta_loca = vector_substract(loca, obj_instance.func_provide_plot_on_obj(integrated_var, **kwargs))
    tempres = cross_product(difference_arc, delta_loca,
                            scale=np.pow(dot_product(delta_loca, delta_loca), -3/2)
                            )
    return scale_product(tempres, permea_of_vaccum/(4*math.pi))

class EMdynamic(EMobj):
    def __init__(self, position=posi_default, velocity=velo_default, acceleration=accl_default, mass=0):
        EMobj.__init__(self, position)
        self.velo = velocity
        self.accl = acceleration
        self.mass = mass


class DynamicParticle(EMdynamic):
    def __init__(self,
                 position_x = posi_default[0], position_y = posi_default[1], position_z = posi_default[2],
                 velocity_x = velo_default[0], velocity_y = velo_default[1], velocity_z = velo_default[2],
                 acceleration_x = accl_default[0], acceleration_y = accl_default[1], acceleration_z = accl_default[2],
                 charge = proton_charge, mass=0,
                 isCountEField = False):
        position = (position_x, position_y, position_z)
        velocity = (velocity_x, velocity_y, velocity_z)
        acceleration = (acceleration_x, acceleration_y, acceleration_z)
        EMdynamic.__init__(self, position, velocity, acceleration, mass)
        self.charge = charge
        self.isCountEField = isCountEField
        self._elec_accl = self.accl
        self._magn_accl = (0, 0, 0)

    def __str__(self):
        return(EMobj.__str__(self) + '| velocity: {0} | acceleration: {1} | charge: {2} | mass: {3}'.format(
            self.velo, self.accl, self.charge, self.mass))
    def indiv_measure_Efield(self, loca, cut_threshold=50):
        if self.isCountEField:
            difference_loca = vector_substract(loca, self.position)
            distance_square = dot_product(difference_loca, difference_loca)
            if distance_square < 1e-16:
                return (0, 0, 0)
            else:
                return scale_product(difference_loca,
                                     self.charge * pow(distance_square, -3/2)/
                                     (4 * math.pi * permit_of_vacuum))
        else:
            return (0, 0, 0)
    def indiv_measure_Mfield(self, loca, cut_threshold=50):
        if self.isCountEField:
            return (0, 0, 0)
        else:
            return (0, 0, 0)

    def calculate_next_dyna_status(self, steplength, do_update=True):
        ele_field_now = total_measure_Efield(self.position)
        mag_field_now = total_measure_Mfield(self.position)
        force_received_ele = scale_product(ele_field_now, self.charge)
        force_received_mag = cross_product(self.velo, mag_field_now, self.charge)

        # print(self._magn_accl, self._elec_accl)

        #Algorithm: open to be altered
        mag_accl = scale_product(force_received_mag, 1 / self.mass)
        ele_accl = scale_product(force_received_ele, 1 / self.mass)
        res_accl = vector_add(mag_accl, ele_accl)
        current_velo_square = dot_product(self.velo, self.velo)
        # print(dot_product(self.velo, self._magn_accl))

        temp_resvelo = vector_add(self.velo, scale_product(self._magn_accl, steplength))
        res_velo = vector_add(scale_product(temp_resvelo,
                                 np.sqrt((current_velo_square) / dot_product(temp_resvelo, temp_resvelo))
                                     # current_velo_square +
                                     # dot_product(self._magn_accl, self._magn_accl) * np.pow(steplength, 2)

                                 ),
                              scale_product(self._elec_accl, steplength))
        res_position = vector_add(self.position,
                                  scale_product(self.velo, steplength),
                                  scale_product(self.accl, math.pow(steplength, 2)/2))
        if do_update:
            self.accl, self.velo, self.charge = (
                res_accl, res_velo, res_position)
        self._elec_accl, self._magn_accl = ele_accl, mag_accl
        return res_accl, res_velo, res_position

class FacingVectorZeroError(Exception): pass
class ChargedWire(EMstatic):
    def __init__(self,
                 position_x=posi_default[0], position_y=posi_default[1], position_z=posi_default[2],
                 facing_x=facing_default[0], facing_y=facing_default[1], facing_z=facing_default[2],
                 lnr_charge_den=0,
                 current=0):

        position = (position_x, position_y, position_z)
        facing = (facing_x, facing_y, facing_z)
        EMstatic.__init__(self, position, facing)
        self.lnr_charge_den = lnr_charge_den
        self.current = current
        if facing == (0, 0, 0):
            raise FacingVectorZeroError

        print(self.__dict__)
#     def __del__(self):
#         tempstr = str(self)
# #        tempobjID = self.objID
# #        print(dict_existed_objs)#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!ddebug code
# #        del dict_existed_objs[tempobjID]
#         print('removed object:' + tempstr)
#         global count_existed_objs
#         count_existed_objs -= 1
    def __str__(self):
        """
        objid: 5; typename: 16; location: 40; facing: 27; other properties: 10
        :return:
        """
        return (EMobj.__str__(self) +
                '| facing: {0} | linear charge density: {1} | current: {2} '.format(
                    self.facing, self.lnr_charge_den, self.current
                ))
    def measure_distance(self, loca, do_return_sqrt=True):
        difference = measure_loca_difference(self.position, loca)
        tempres = cross_product(difference, self.facing)
        # print(dot_product(tempres, tempres) / dot_product(self.facing, self.facing))
        if do_return_sqrt:
            return np.sqrt(
                dot_product(tempres, tempres) / dot_product(self.facing, self.facing)
            )
        else:
            return dot_product(tempres, tempres) / dot_product(self.facing, self.facing)

    def indiv_measure_Efield(self, loca, cut_threshold=50):
        difference = measure_loca_difference(self.position, loca)
        tempres = cross_product(difference, self.facing)
        res = cross_product(self.facing, tempres,
                             scale = self.lnr_charge_den /
                                     (2 * math.pi * permit_of_vacuum * dot_product(self.facing, self.facing) *
                             self.measure_distance(loca, do_return_sqrt=False)))
        # if dot_product(res, res) > cut_threshold * cut_threshold:
        #     return (0, 0, 0)
        # else:
        return res
    def indiv_measure_Mfield(self, loca, cut_threshold=50):
        tempres = cross_product(measure_loca_difference(self.position, loca), self.facing)
        tempscale = ((permea_of_vaccum * self.current * np.sqrt(dot_product(self.facing, self.facing))) /
                     (2 * math.pi * dot_product(tempres, tempres)))
        res = scale_product(tempres, -1 * tempscale)
        # if dot_product(res, res) > cut_threshold * cut_threshold:
        #     return (0, 0, 0)
        # else:
        return res

# def provide_plot_on_ring(ring_facing, ring_location, radius, angle):
#     benchvec1, benchvec2, benchvec3 = single_schmitt(ring_facing)
#     tempres = vector_add(scale_product(benchvec2, math.cos(angle)),
#                          scale_product(benchvec3, math.sin(angle)))
#     return vector_add(ring_location,
#                       scale_product(tempres, radius))
var_piece_default = 1802
class GalvanizedRing(EMstatic):
    def __init__(self,
                 position_x=posi_default[0], position_y=posi_default[1], position_z=posi_default[2],
                 facing_x=facing_default[0], facing_y=facing_default[1], facing_z=facing_default[2],
                 radius=1.0, lnr_charge_den=0.0, current=0.0):
        position = (position_x, position_y, position_z)
        facing = (facing_x, facing_y, facing_z)
        EMstatic.__init__(self, position, facing)
        self.radius = radius
        self.current = current
        self.lnr_charge_den = lnr_charge_den
        if facing == (0, 0, 0):
            raise FacingVectorZeroError
        self._benchvec1, self._benchvec2, self._benchvec3 = single_schmitt(self.facing)
    def __str__(self):
        return (EMobj.__str__(self) +
                '| facing: {0} | linear charge density: {1} | current: {2} | radius: {3}'.format(
                    self.facing, self.lnr_charge_den, self.current, self.radius
                ))
    def func_provide_diffrnt_plot(self, angle, loca):
        temp_difference_arc = vector_add(scale_product(self._benchvec2, -1*self.radius*np.sin(angle)),
                                    scale_product(self._benchvec3, self.radius*np.cos(angle)))
        return scale_product(temp_difference_arc, self.current)
    def func_provide_plot_on_obj(self, angle, loca):
        tempres = vector_add(scale_product(self._benchvec2, np.cos(angle)),
                             scale_product(self._benchvec3, np.sin(angle)))
        return scale_product(vector_add(tempres, self.position), self.radius)
    def indiv_measure_Mfield(self, loca, var_pieces=var_piece_default, cut_threshold=50):
        res = line_vector_integrate(integrated_func, 0,
                                     obj_instance=self, loca=loca,
                                     var_pieces=var_pieces)
        # if dot_product(res, res) > cut_threshold * cut_threshold:
        #     return (0, 0, 0)
        # else:
        return res
    def indiv_measure_Efield(self, loca, cut_threshold=50): return (0, 0, 0)

class ChargedConductSphere(EMstatic):
    def __init__(self,
                 position_x=posi_default[0], position_y=posi_default[1], position_z=posi_default[2],
                 facing_x=facing_default[0], facing_y=facing_default[1], facing_z=facing_default[2],
                 radius=0,
                 total_charge=0,
                 current=0
                 ):
        position = (position_x, position_y, position_z)
        facing = (facing_x, facing_y, facing_z)
        EMstatic.__init__(self, position, facing)
        self.total_charge = total_charge
        self.current = current
        self.radius = radius
        if facing == (0, 0, 0):
            raise FacingVectorZeroError
        print(self.__dict__)
    def __str__(self):
        return (EMobj.__str__(self) +
                '| facing: {0} | charge: {1} | radius: {2} '.format(
                    self.facing, self.total_charge, self.radius
                ))
    def measure_distance(self, loca, do_return_square=False):
        difference = measure_loca_difference(self.position, loca)
        if not do_return_square:
            return np.sqrt(dot_product(difference, difference))
        else:
            return dot_product(difference, difference)
    def indiv_measure_Efield(self, loca, cut_threshold=50):
        if self.measure_distance(loca) < self.radius:
            res = (0, 0, 0)
        elif self.measure_distance(loca) > self.radius:
            res = scale_product(measure_loca_difference(self.position, loca),
                                 self.total_charge / (
                                 4 * math.pi * permit_of_vacuum * np.pow(
                                 self.measure_distance(loca), 3)))
        else:
            res = scale_product(measure_loca_difference(self.position, loca),
                                 self.total_charge / (
                                 8 * math.pi * permit_of_vacuum * np.pow(
                                 self.radius, 3)))

        # if dot_product(res, res) > cut_threshold * cut_threshold:
        #     return (0, 0, 0)
        # else:
        return res
    def indiv_measure_Mfield(self, loca, cut_threshold=50):
        return (0, 0, 0)

class UniEField(EMstatic):
    def __init__(self,
                 judger,
                 position_x=posi_default[0], position_y=posi_default[1], position_z=posi_default[2],
                 facing_x=facing_default[0], facing_y=facing_default[1], facing_z=facing_default[2],
                 EFIntensity=0):
        # assert (border_x > posi_default[0]
        #         and border_y > posi_default[1]
        #         and border_z > posi_default[2]), \
        #     "Coordinates of border must greater than that of position"
        position = (position_x, position_y, position_z)
        facing = (facing_x, facing_y, facing_z)
        EMstatic.__init__(self, position, facing)
        self.judger = judger #judger: instance.judger(position) -> Boolean, position: tuple
#        self.shape = shape
        self.EFIntensity = EFIntensity

    def indiv_measure_Efield(self, loca, cut_threshold=50): #loca: tuple of location of the point measured
        if self.judger(loca):
            reducer = math.sqrt(math.pow(self.facing[0], 2) +
                                math.pow(self.facing[1], 2) +
                                math.pow(self.facing[2], 2))
            return (self.EFIntensity * self.facing[0] / reducer,
                    self.EFIntensity * self.facing[1] / reducer,
                    self.EFIntensity * self.facing[2] / reducer)
        else:
            return (0, 0, 0)

    def indiv_measure_Mfield(self, loca, cut_threshold): return (0, 0, 0)

class CubicUniEField(UniEField):
    def __init__(self,
                 position_x = posi_default[0], position_y = posi_default[1], position_z = posi_default[2],
                 border_x = posi_default[0], border_y = posi_default[1], border_z = posi_default[2],
                 facing_x = facing_default[0], facing_y = facing_default[1], facing_z = facing_default[2],
                 EFIntensity = 0):
        def cubic_judger(loca):
            return (position_x <= loca[0] <= border_x
                    and position_y <= loca[1] <= border_y
                    and position_z <= loca[2] <= border_z)
        UniEField.__init__(self, cubic_judger,
                           position_x, position_y, position_z,
                           facing_x, facing_y, facing_z,
                           EFIntensity)
        self.border = (border_x, border_y, border_z)

    def __str__(self):
        return ('{0:>5} | Uniform Electr Field | range: ({1}~{4}, {2}~{5}, {3}~{6}) | Cubic '
                '| Field Intensity: {7} | Field Direction : {8}').format(
            self.objID,
            *self.position,
            *self.border,
            self.EFIntensity,
            self.facing
        )
class InfiniteUniEField(UniEField):
    def __init__(self,
                 position_x = posi_default[0], position_y = posi_default[1], position_z = posi_default[2],
                 facing_x = facing_default[0], facing_y = facing_default[1], facing_z = facing_default[2],
                 EFIntensity = 0):
        def infinite_judger(loca):
            return True
        UniEField.__init__(self, infinite_judger,
                           position_x, position_y, position_z,
                           facing_x, facing_y, facing_z,
                           EFIntensity)
    def __str__(self):
        return ('{0:>5} | Uniform Electr Field | Infinite '
                '| Field Intensity: {1} | Field Direction: {2}').format(
            self.objID,
            self.EFIntensity,
            self.facing
        )

class UniMField(EMstatic):
    def __init__(self,
                 judger,
                 position_x=posi_default[0], position_y=posi_default[1], position_z=posi_default[2],
                 facing_x=facing_default[0], facing_y=facing_default[1], facing_z=facing_default[2],
                 MFIntensity=0):
        # assert (border_x > posi_default[0]
        #         and border_y > posi_default[1]
        #         and border_z > posi_default[2]), \
        #     "Coordinates of border must greater than that of position"
        position = (position_x, position_y, position_z)
        facing = (facing_x, facing_y, facing_z)
        EMstatic.__init__(self, position, facing)
        self.judger = judger #judger: instance.judger(position) -> Boolean, position: tuple
#        self.shape = shape
        self.MFIntensity = MFIntensity

    def indiv_measure_Efield(self, loca, cut_threshold=50): return (0, 0, 0)
    def indiv_measure_Mfield(self, loca, cut_threshold=50): #loca: tuple of location of the point measured
        if self.judger(loca):
            reducer = math.sqrt(math.pow(self.facing[0], 2) +
                                math.pow(self.facing[1], 2) +
                                math.pow(self.facing[2], 2))
            return (self.MFIntensity * self.facing[0] / reducer,
                    self.MFIntensity * self.facing[1] / reducer,
                    self.MFIntensity * self.facing[2] / reducer)
        else:
            return (0, 0, 0)

class CubicUniMField(UniMField):
    def __init__(self,
                 position_x = posi_default[0], position_y = posi_default[1], position_z = posi_default[2],
                 border_x = posi_default[0], border_y = posi_default[1], border_z = posi_default[2],
                 facing_x = facing_default[0], facing_y = facing_default[1], facing_z = facing_default[2],
                 MFIntensity = 0):
        def cubic_judger(loca):
            return (position_x <= loca[0] <= border_x
                    and position_y <= loca[1] <= border_y
                    and position_z <= loca[2] <= border_z)
        UniMField.__init__(self, cubic_judger,
                           position_x, position_y, position_z,
                           facing_x, facing_y, facing_z,
                           MFIntensity)
        self.border = (border_x, border_y, border_z)

    def __str__(self):
        return '{0:>5} | Uniform Magnet Field | range: ({1}~{4}, {2}~{5}, {3}~{6}) | Cubic | Field Intensity: {7}'.format(
            self.objID,
            *self.position,
            *self.border,
            self.MFIntensity
        )
class InfiniteUniMField(UniMField):
    def __init__(self,
                 position_x = posi_default[0], position_y = posi_default[1], position_z = posi_default[2],
                 facing_x = facing_default[0], facing_y = facing_default[1], facing_z = facing_default[2],
                 MFIntensity = 0):
        def infinite_judger(loca):
            return True
        UniMField.__init__(self, infinite_judger,
                           position_x, position_y, position_z,
                           facing_x, facing_y, facing_z,
                           MFIntensity)
    def __str__(self):
        return ('{0:>5} | Uniform Magnet Field | Infinite '
                '| Field Intensity: {1} | Field Direction: {2}').format(
            self.objID,
            self.MFIntensity,
            self.facing
        )

if __name__ == '__main__':

    # b = GalvanizedRing(0, 0, 0, 0, 0, 1, 1, 1e7, 1)
    # print(b.indiv_measure_Mfield((0, 0, -1), var_pieces=36002)) #BEST

    c = ChargedWire(0, 0, 0, 0, 0, 1, 1e-12, 1)
    p = DynamicParticle(0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1)
