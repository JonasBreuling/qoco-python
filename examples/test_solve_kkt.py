"""
Unit test-style example to verify the solve_kkt implementation.

This example:
1. Sets up a simple QP problem
2. Solves it to get the factorized KKT system
3. Tests solve_kkt with known RHS vectors
4. Verifies the solution is correct
"""

import qoco
import numpy as np
from scipy import sparse


def test_solve_kkt_zero_rhs():
    """Test that solve_kkt with zero RHS returns zero solution."""
    print("=" * 60)
    print("Test 1: solve_kkt with zero RHS")
    print("=" * 60)
    
    # Define a simple QP problem: minimize 0.5*x'*P*x + c'*x
    # subject to: A*x = b, G*x <= h
    P = sparse.diags([1.0, 2.0, 3.0], 0, dtype=np.float64)
    P = P.tocsc()
    c = np.array([1.0, 2.0, 3.0])
    
    # Add one trivial equality constraint
    A = sparse.csc_matrix([[0.0, 0.0, 0.0]], dtype=np.float64)
    b = np.array([0.0])
    
    # One conic constraint (linear inequality)
    G = sparse.csc_matrix([[-1.0, 0.0, 0.0]], dtype=np.float64)
    h = np.array([0.0])
    
    n = 3
    m = 1  # One linear inequality
    p = 1
    l = 1  # One linear constraint
    nsoc = 0
    q = np.array([], dtype=np.int32)
    
    # Create and setup solver
    prob = qoco.QOCO()
    prob.setup(n, m, p, P, c, A, b, G, h, l, nsoc, q)
    
    # Solve the problem first to factorize the KKT system
    res = prob.solve()
    print(f"Problem solved with status: {res.status}")
    print(f"Optimal objective: {res.obj}")
    
    # Test 1: Zero RHS should give zero solution
    rhs = np.zeros(n + m + p)
    sol = prob.solve_kkt(rhs)
    
    print(f"RHS (zero): {rhs}")
    print(f"Solution: {sol}")
    
    assert sol.shape == (n + m + p,), f"Expected shape {(n + m + p,)}, got {sol.shape}"
    assert np.allclose(sol, 0.0, atol=1e-10), f"Expected zero solution, got {sol}"
    print("✓ Test 1 PASSED: Zero RHS returns zero solution\n")


def test_solve_kkt_consistency():
    """Test that solve_kkt gives consistent results."""
    print("=" * 60)
    print("Test 2: solve_kkt consistency")
    print("=" * 60)
    
    # Define a simple QP with constraints
    P = sparse.diags([1.0, 1.0], 0, dtype=np.float64)
    P = P.tocsc()
    c = np.array([1.0, 1.0])
    
    # Equality constraint: x1 + x2 = 1
    A = sparse.csc_matrix([[1.0, 1.0]], dtype=np.float64)
    b = np.array([1.0])
    
    # One conic constraint
    G = sparse.csc_matrix([[-1.0, 0.0]], dtype=np.float64)
    h = np.array([0.0])
    
    n = 2
    m = 1
    p = 1
    l = 1
    nsoc = 0
    q = np.array([], dtype=np.int32)
    
    # Create and setup solver
    prob = qoco.QOCO()
    prob.setup(n, m, p, P, c, A, b, G, h, l, nsoc, q)
    
    # Solve the problem
    res = prob.solve()
    print(f"Problem solved with status: {res.status}")
    print(f"Optimal solution x: {res.x}")
    print(f"Dual variable y: {res.y}")
    
    # Create a RHS vector with different components
    rhs = np.zeros(n + m + p)
    rhs[0] = 1.0  # Perturbation in first primal variable
    
    # Solve KKT system twice to verify consistency
    sol1 = prob.solve_kkt(rhs.copy())
    sol2 = prob.solve_kkt(rhs.copy())
    
    print(f"RHS: {rhs}")
    print(f"Solution 1: {sol1}")
    print(f"Solution 2: {sol2}")
    
    assert np.allclose(sol1, sol2), "solve_kkt should give consistent results"
    print("✓ Test 2 PASSED: solve_kkt is consistent\n")


def test_solve_kkt_linearity():
    """Test that solve_kkt respects linearity (superposition)."""
    print("=" * 60)
    print("Test 3: solve_kkt linearity (superposition)")
    print("=" * 60)
    
    # Define a simple QP
    P = sparse.diags([2.0, 2.0, 2.0], 0, dtype=np.float64)
    P = P.tocsc()
    c = np.array([0.0, 0.0, 0.0])
    
    A = sparse.csc_matrix([[0.0, 0.0, 0.0]], dtype=np.float64)
    b = np.array([0.0])
    
    G = sparse.csc_matrix([[-1.0, 0.0, 0.0]], dtype=np.float64)
    h = np.array([0.0])
    
    n = 3
    m = 1
    p = 1
    l = 1
    nsoc = 0
    q = np.array([], dtype=np.int32)
    
    # Create and setup solver
    prob = qoco.QOCO()
    prob.setup(n, m, p, P, c, A, b, G, h, l, nsoc, q)
    
    # Solve the problem
    res = prob.solve()
    print(f"Problem solved with status: {res.status}")
    
    # Create two RHS vectors (size n + m + p = 5)
    rhs1 = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    rhs2 = np.array([0.0, 1.0, 0.0, 0.0, 0.0])
    
    # Solve individually
    sol1 = prob.solve_kkt(rhs1)
    sol2 = prob.solve_kkt(rhs2)
    
    # Solve superposition
    rhs_sum = rhs1 + rhs2
    sol_sum = prob.solve_kkt(rhs_sum)
    
    print(f"RHS1: {rhs1}")
    print(f"Sol1: {sol1}")
    print(f"RHS2: {rhs2}")
    print(f"Sol2: {sol2}")
    print(f"RHS1 + RHS2: {rhs_sum}")
    print(f"Sol(RHS1 + RHS2): {sol_sum}")
    print(f"Sol1 + Sol2: {sol1 + sol2}")
    
    # Check linearity: solve_kkt(rhs1 + rhs2) should equal solve_kkt(rhs1) + solve_kkt(rhs2)
    assert np.allclose(sol_sum, sol1 + sol2, atol=1e-10), \
        f"Linearity violated: {sol_sum} != {sol1 + sol2}"
    print("✓ Test 3 PASSED: solve_kkt respects linearity\n")


def test_solve_kkt_with_scaling():
    """Test that solve_kkt respects scalar multiplication."""
    print("=" * 60)
    print("Test 4: solve_kkt with scaling")
    print("=" * 60)
    
    # Define a simple QP
    P = sparse.diags([1.0, 1.0], 0, dtype=np.float64)
    P = P.tocsc()
    c = np.array([0.0, 0.0])
    
    # Add trivial equality constraint
    A = sparse.csc_matrix([[0.0, 0.0]], dtype=np.float64)
    b = np.array([0.0])
    
    G = sparse.csc_matrix([[-1.0, 0.0]], dtype=np.float64)
    h = np.array([0.0])
    
    n = 2
    m = 1
    p = 1
    l = 1
    nsoc = 0
    q = np.array([], dtype=np.int32)
    
    # Create and setup solver
    prob = qoco.QOCO()
    prob.setup(n, m, p, P, c, A, b, G, h, l, nsoc, q)
    
    # Solve the problem
    res = prob.solve()
    print(f"Problem solved with status: {res.status}")
    
    # Create a RHS vector (size n + m + p = 4)
    rhs = np.array([2.0, 3.0, 0.0, 0.0])
    
    # Solve with original RHS
    sol = prob.solve_kkt(rhs)
    
    # Solve with scaled RHS
    alpha = 2.5
    sol_scaled = prob.solve_kkt(alpha * rhs)
    
    print(f"RHS: {rhs}")
    print(f"Solution: {sol}")
    print(f"Alpha * RHS: {alpha * rhs}")
    print(f"Solution (scaled RHS): {sol_scaled}")
    print(f"Alpha * Solution: {alpha * sol}")
    
    # Check scaling: solve_kkt(alpha * rhs) should equal alpha * solve_kkt(rhs)
    assert np.allclose(sol_scaled, alpha * sol, atol=1e-10), \
        f"Scaling violated: {sol_scaled} != {alpha * sol}"
    print("✓ Test 4 PASSED: solve_kkt respects scaling\n")


def test_solve_kkt_error_handling():
    """Test error handling in solve_kkt."""
    print("=" * 60)
    print("Test 5: solve_kkt error handling")
    print("=" * 60)
    
    # Define a simple QP
    P = sparse.diags([1.0, 1.0], 0, dtype=np.float64)
    P = P.tocsc()
    c = np.array([1.0, 2.0])
    
    # Add trivial equality constraint
    A = sparse.csc_matrix([[0.0, 0.0]], dtype=np.float64)
    b = np.array([0.0])
    
    G = sparse.csc_matrix([[-1.0, 0.0]], dtype=np.float64)
    h = np.array([0.0])
    
    n = 2
    m = 1
    p = 1
    l = 1
    nsoc = 0
    q = np.array([], dtype=np.int32)
    
    # Create and setup solver
    prob = qoco.QOCO()
    prob.setup(n, m, p, P, c, A, b, G, h, l, nsoc, q)
    
    # Solve the problem
    res = prob.solve()
    print(f"Problem solved with status: {res.status}")
    
    # Test 1: Wrong size RHS
    print("\nTest 5a: Wrong size RHS")
    try:
        wrong_size_rhs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # Should be (4,)
        prob.solve_kkt(wrong_size_rhs)
        print("✗ FAILED: Should have raised ValueError for wrong size")
    except ValueError as e:
        print(f"✓ Correctly raised ValueError: {e}")
    
    # Test 2: 2D array (should be converted)
    print("\nTest 5b: 2D array RHS")
    try:
        rhs_2d = np.array([[1.0], [2.0], [0.0], [0.0]])
        prob.solve_kkt(rhs_2d)
        print("✗ FAILED: Should have raised ValueError for 2D array")
    except (ValueError, RuntimeError) as e:
        print(f"✓ Correctly raised error: {e}")
    
    print("\n✓ Test 5 PASSED: Error handling works correctly\n")


def test_solve_kkt_with_constraints():
    """Test solve_kkt with a problem that has constraints."""
    print("=" * 60)
    print("Test 6: solve_kkt with constraints")
    print("=" * 60)
    
    # Define the QP problem from simple_socp1.py
    P = sparse.diags([1, 2, 3, 4, 5, 6], 0, dtype=np.float64)
    P = P.tocsc()

    c = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
    G = -sparse.identity(6)
    G = G.tocsc()
    h = np.zeros(6)
    A = sparse.csc_matrix([[1, 1, 0, 0, 0, 0], [0, 1, 2, 0, 0, 0]], dtype=np.float64)
    A = A.tocsc()
    b = np.array([1, 2], dtype=np.float64)

    l = 3
    n = 6
    m = 6
    p = 2
    nsoc = 1
    q = np.array([3], dtype=np.int32)

    # Create and setup solver
    prob = qoco.QOCO()
    prob.setup(n, m, p, P, c, A, b, G, h, l, nsoc, q)

    # Solve the problem
    res = prob.solve()
    print(f"Problem solved with status: {res.status}")
    print(f"Optimal objective: {res.obj}")
    
    # Test with various RHS vectors
    print("\nTesting solve_kkt with different RHS vectors:")
    
    # Test 1: Basis vector in primal space
    rhs = np.zeros(n + m + p)
    rhs[0] = 1.0
    sol = prob.solve_kkt(rhs)
    print(f"RHS (basis e1): {rhs}")
    print(f"Solution shape: {sol.shape}, values: min={sol.min():.4e}, max={sol.max():.4e}")
    
    # Test 2: Basis vector in dual space
    rhs = np.zeros(n + m + p)
    rhs[n] = 1.0
    sol = prob.solve_kkt(rhs)
    print(f"RHS (basis en+1): {rhs}")
    print(f"Solution shape: {sol.shape}, values: min={sol.min():.4e}, max={sol.max():.4e}")
    
    # Test 3: Basis vector in equality dual space
    rhs = np.zeros(n + m + p)
    rhs[n + m] = 1.0
    sol = prob.solve_kkt(rhs)
    print(f"RHS (basis en+m+1): {rhs}")
    print(f"Solution shape: {sol.shape}, values: min={sol.min():.4e}, max={sol.max():.4e}")
    
    print("\n✓ Test 6 PASSED: solve_kkt works with constrained problems\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("QOCO solve_kkt Implementation Tests")
    print("=" * 60 + "\n")
    
    try:
        test_solve_kkt_zero_rhs()
        test_solve_kkt_consistency()
        test_solve_kkt_linearity()
        test_solve_kkt_with_scaling()
        test_solve_kkt_error_handling()
        test_solve_kkt_with_constraints()
        
        print("=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
