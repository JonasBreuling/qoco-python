import qoco
import numpy as np
from scipy import sparse


if __name__ == "__main__":
    # Define problem data
    P = sparse.diags([1, 2, 3, 4, 5, 6], 0, dtype=None)
    P = P.tocsc()

    c = np.array([1, 2, 3, 4, 5, 6], dtype=float)
    G = -sparse.identity(6)
    G = G.tocsc()
    h = np.zeros(6)
    A = sparse.csc_matrix([[1, 1, 0, 0, 0, 0], [0, 1, 2, 0, 0, 0]])
    A = A.tocsc()
    b = np.array([1, 2])

    l = 3
    n = 6
    m = 6
    p = 2
    nsoc = 1
    q = np.array([3], dtype=float)

    # Create an QOCO object.
    prob = qoco.QOCO()

    # Setup workspace.
    prob.setup(n, m, p, P, c, A, b, G, h, l, nsoc, q)

    # Solve problem.
    res = prob.solve()

    opt_obj = 4.042
    assert res.status == "QOCO_SOLVED"
    assert abs(res.obj - opt_obj) <= 1e-4

    # solve regularized KKT system with the existing LDL factorization
    rhs = np.ones(prob.n + prob.m + prob.p)
    sol = prob.solve_kkt(rhs)

    # multiply original KKT matrix by some vector
    product = prob.kkt_multiply(sol)
    assert np.allclose(product, rhs, atol=1e-8)

    # print("res.x:", res.x)
    print("sol:", sol)
    print("KKT @ sol:", product)
    print("res:", res)

    # possibly update settings
    prob.update_settings(max_iters=1000, tol=1e-8)

    # update vector data and solve KKT system again
    # (None means that the corresponding data is not updated)
    prob.update_vector_data(c=None, h=h + 1, b=None)
    res = prob.solve()
    print("res:", res)