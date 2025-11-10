"""
This module wraps around the ytopt generator.
"""
import numpy as np
from libensemble.message_numbers import STOP_TAG, PERSIS_STOP, FINISHED_PERSISTENT_GEN_TAG, EVAL_GEN_TAG
from libensemble.tools.persistent_support import PersistentSupport


__all__ = ['persistent_ytopt']


def persistent_ytopt(H, persis_info, gen_specs, libE_info):

    ps = PersistentSupport(libE_info, EVAL_GEN_TAG)
    user_specs = gen_specs['user']
    ytoptimizer = user_specs['ytoptimizer']

    tag = None
    calc_in = None
    first_call = True
    first_write = True
    fields = [i[0] for i in gen_specs['out']]
    # The order of fields must semantically match what Ytopt will see,
    # else Ytopt will fail to match any key - eval pairings.
    #
    # Due to brittleness in libEnsemble interactions, reordering
    # this 'sometimes-present' field to the back minimizes changes.
    # In the future, enforced key ordering for Ytopt's evaluation keys
    # should eliminate need for this adjustment.
    fields.pop(fields.index('p3'))
    fields.append('p3')

    # Send batches until manager sends stop tag
    while tag not in [STOP_TAG, PERSIS_STOP]:

        if first_call:
            ytopt_points = ytoptimizer.ask_initial(n_points=user_specs['num_sim_workers'])  # Returns a list
            batch_size = len(ytopt_points)
            first_call = False
        else:
            batch_size = len(calc_in)
            results = []
            for entry in calc_in:
                field_params = {}
                for field in fields:
                    # Presence fields are NOT passed to Ytopt.
                    # However, they must be provided to LibEnsemble due to API semantics.
                    # We represent 'not-present' values with np.nan for scikit-optimize.
                    if "present" in field:
                        continue
                    elif field+"_present" in fields:
                        # This is the from->LibEnsemble sentinel adjustment
                        if not entry[fields.index(field+"_present")]:
                            field_params[field] = np.nan
                            continue
                        else:
                            if field not in entry.dtype.names:
                                continue
                            field_params[field] = float(entry[field][0])
                    else:
                        if field not in entry.dtype.names:
                            continue
                        field_params[field] = entry[field][0]
                results += [(field_params, entry['objective'])]
            print('results: ', results)
            ytoptimizer.tell(results)

            ytopt_points = ytoptimizer.ask(n_points=batch_size)  # Returns a generator that we convert to a list
            ytopt_points = list(ytopt_points)[0]
        # We must evaluate the presence here, as LibEnsemble will not permit us to assing
        # np.nan as an integer parameter value.
        # We'll use -1 (invalid from ConfigSpace POV) as a sentinel, and restore the nan-ness after
        # LibEnsemble has done its handoff.
        # This presence field exists to indicate that such restoration will occur. The sentinel
        # does not need to be invalid for the ConfigSpace, but doing so helps validate that this
        # wrapper intercepts its own meddling before propagating it to components that should not
        # be affected.
        for point in ytopt_points:
            point['p3_present'] = not np.isnan(point['p3'])
        # The hand-off of information from ytopt to libE is below. This hand-off may be brittle.
        H_o = np.zeros(batch_size, dtype=gen_specs['out'])
        for i, entry in enumerate(ytopt_points):
            for key, value in entry.items():
                # This is the to->LibEnsemble sentinel adjustment.
                # String data and many other representations cannot be checked by np.isnan.
                # If the data isn't numerical, we're not making a nan-based adjustment so we
                # just assign the original value.
                try:
                    if np.isnan(value):
                        H_o[i][key] = -1
                    else:
                        H_o[i][key] = value
                except:
                    H_o[i][key] = value

        # This returns the requested points to the libE manager, which will
        # perform the sim_f evaluations and then give back the values.
        tag, Work, calc_in = ps.send_recv(H_o)
        print('received:', calc_in, flush=True)

        if calc_in is not None and len(calc_in):
            b = []
            for calc_result in calc_in:
                for entry in calc_result:
                    # Most entries should be 1-length np.ndarrays, however if the top-level implementer
                    # failed to indicate to LibEnsemble that this would be the case, they'll come back
                    # as just the value.
                    try:
                        b += [str(entry[0])]
                    except:
                        b += [str(entry)]

                with open('../../results.csv', 'a') as f:
                    if first_write:
                        f.write(",".join(calc_result.dtype.names)+ "\n")
                        first_write = False
                    f.write(",".join(b)+ "\n")

    return H_o, persis_info, FINISHED_PERSISTENT_GEN_TAG

