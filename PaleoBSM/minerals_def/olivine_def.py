from .construct_minerals import Minerals
# Create new mineral object
olivineObj = Minerals('Olivine', 4)

# paths and values
raw_data_path_array = ['PaleoBSM/SRIM_raw_data/Olivine/O/', \
    'PaleoBSM/SRIM_raw_data/Olivine/Si/', \
    'PaleoBSM/SRIM_raw_data/Olivine/Fe/', \
    'PaleoBSM/SRIM_raw_data/Olivine/Mg/']


derived_data_path_array = ['PaleoBSM/SRIM_derived_data/Olivine/O/', \
    'PaleoBSM/SRIM_derived_data/Olivine/Si/', \
    'PaleoBSM/SRIM_derived_data/Olivine/Fe/', \
    'PaleoBSM/SRIM_derived_data/Olivine/Mg/']

Er_avail_array = [0.01,0.03,0.05,0.07,0.1, 0.2, 0.3, 0.4, 0.5, \
                           5, 8, 10, 15, 18, 23, 25, 30, 33, 38, 40]

spin_array = [5/2, 1/2, 1/2, 5/2]
spin_isotopic_fractions_array = [0.00048, 0.0467, 0.021, 0.1] # isotopic abundance for the nuclear spin listed in spin_array

alpha = [[-0.09428571428571428, 2.514102564102564, -10.5], \
         [-0.0639795918367347, 2.596153846153846, -10.5], \
            [-0.15020408163265306, 1.0218461538461538, 0.005263157894736831], \
                [-0.05183673469387755, 2.626923076923077, -10.947368421052632]]

mean = [[-0.0005306238348739585, 0.022600608190676445, 3.2715588249339556, -1.1363484483823816], \
        [0.0006507613414695835, -0.013618419650077224, 1.1808575952780669, -0.15418383405102157], \
        [0.00019113789153530844, -0.01317741886686746, 0.7803281530132474, -0.0014395486718311488], \
        [0.0007708657798528288, -0.021812513981345316, 1.574071910279722, -0.3762470479337966]]

var = [[-9.435880183523317e-05, 0.000466517183862611, 1.0907097943850064, 0.8618865470849618], \
       [0.000510735235186759, -0.03687721320637635, 1.1641849263411248, 0.45916912857533126], \
       [0.00013214703600348166,-0.01190037788836669, 0.6459426710739214, 0.4582531443067176], \
        [0.00020623016285632116, -0.024790310304098027, 1.3112367619452836, 0.6079615098527893]]


# set mineral attributes
olivineObj.set_composition(['O', 'Si', 'Fe', 'Mg'])
olivineObj.set_atomic_number([8, 14, 26, 12])
olivineObj.set_atomic_masses([15.999, 28.0855, 55.845, 24.305])
olivineObj.set_atomic_fractions([0.41, 0.18, 0.14, 0.25])
olivineObj.set_number_fractions([4, 1, 0.39771717171717175, 2-0.39771717171717175])
olivineObj.set_raw_data_paths(raw_data_path_array)
olivineObj.set_derived_data_paths(derived_data_path_array)
olivineObj.set_Er_avail(Er_avail_array)
olivineObj.set_nuclear_spins(spin_array)
olivineObj.set_spin_isotopic_fractions(spin_isotopic_fractions_array)
olivineObj.set_alpha_params(alpha)
olivineObj.set_mean_params(mean)
olivineObj.set_var_params(var)


print(olivineObj.name)