import numpy as np

from pycondest import cond1est, _available_methods


def test_random_matrix(test_matrix_data):
    data = test_matrix_data
    for method in _available_methods:
        cond1 = cond1est(data["A"], method=method)
        if method == "hager":
            rtol = 1e-10
            atol = 0
            cond1_octave = float(data["cond1_octave"].flatten()[0])
            np.testing.assert_allclose(
                cond1_octave, cond1, rtol=rtol, atol=atol
            )
            if "cond1_dense" in data:
                cond1_dense = float(data["cond1_dense"].flatten()[0])
                isclose = np.allclose(cond1_dense, cond1, rtol=rtol, atol=atol)
                if not isclose:
                    cond1_spg = cond1est(data["A"], method="spg")
                    message = (
                        "cond1est is not close to the actual condition number"
                        " obtained from dense linear algebra:\n"
                        f"cond1est(hager):  {cond1:.20e}\n"
                        f"octave:           {cond1_octave:.20e}\n"
                        f"dense:            {cond1_dense:.20e}\n"
                        f"cond1est(spg):    {cond1_spg:.20e}\n"
                    )
                    print(message)
