"""Port of tmapper_tools/CycleCount.m from the MATLAB toolbox.

CycleCount.m is itself a third-party algorithm (not original to the
tmapper toolbox), reproduced here under its original license:

    Authors: P.-L. Giscard, N. Kriege, R. Wilson, Septembre 2016

    Copyright (c) 2017, Pierre-Louis Giscard
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions
    are met:

        * Redistributions of source code must retain the above
          copyright notice, this list of conditions and the following
          disclaimer.
        * Redistributions in binary form must reproduce the above
          copyright notice, this list of conditions and the following
          disclaimer in the documentation and/or other materials
          provided with the distribution.
        * Neither the name of the University of York nor the names of
          its contributors may be used to endorse or promote products
          derived from this software without specific prior written
          permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
    FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
    COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
    INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
    BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
    LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
    ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    POSSIBILITY OF SUCH DAMAGE.
"""

import numpy as np


def cycle_count(A, L0):
    """Count all simple cycles of length up to ``L0`` (inclusive) on a
    graph whose adjacency matrix is ``A``, via the combinatorial-sieve
    algorithm of Giscard, Kriege & Wilson (2016).

    Parameters
    ----------
    A : array_like, shape (N, N)
        Adjacency matrix of the graph (directed or undirected, weighted
        or unweighted).
    L0 : int
        Maximum length of the simple cycles to count.

    Returns
    -------
    numpy.ndarray, shape (L0,)
        ``primes[i]`` is the number of simple cycles of length ``i + 1``,
        for ``i`` in ``0..L0-1``.
    """
    A = np.asarray(A, dtype=float).copy()
    L0 = int(L0)

    primes = np.zeros(L0, dtype=complex)
    primes[0] += np.trace(A)  # self-loops (length-1 cycles)
    np.fill_diagonal(A, 0)
    A = _clean_matrix(A)

    directed = not np.array_equal(A, A.T)
    if directed:
        A_undir = ((A != 0) | (A.T != 0)).astype(float)
    else:
        A_undir = (A != 0).astype(float)

    size = A.shape[0]
    if L0 > size:
        L0 = size

    allowed_vert = np.ones(size, dtype=bool)
    for i in range(size - 1):
        allowed_vert[i] = False
        neighbourhood = np.zeros(size)
        neighbourhood[i] = 1
        neighbourhood = neighbourhood + A_undir[i, :]
        primes = _recursive_subgraphs(A, A_undir, L0, [i], allowed_vert, primes, neighbourhood)

    return primes.real


def _recursive_subgraphs(A, A_undir, L0, subgraph, allowed_vert, primes, neighbourhood):
    """Find all connected induced subgraphs of size up to L0 containing
    ``subgraph``, accumulating their contribution into ``primes``."""
    allowed_vert = allowed_vert.copy()  # this frame's mutations stay local

    L = len(subgraph)
    neighbours_number = int(np.count_nonzero(neighbourhood)) - L
    primes = _prime_count(A, L0, subgraph, neighbours_number, primes)

    if L == L0:
        return primes

    neighbours = np.flatnonzero(neighbourhood.astype(bool) & allowed_vert)
    for v in neighbours:
        new_subgraph = subgraph + [int(v)]
        allowed_vert[v] = False  # persists across the rest of this loop
        new_neighbourhood = neighbourhood + A_undir[v, :]
        primes = _recursive_subgraphs(A, A_undir, L0, new_subgraph, allowed_vert, primes, new_neighbourhood)

    return primes


def _prime_count(A, L0, subgraph, neighbours_number, primes):
    """Combinatorial-sieve contribution of one connected induced subgraph."""
    subgraph_size = len(subgraph)
    idx = np.ix_(subgraph, subgraph)
    x = A[idx]

    xeig = np.linalg.eigvals(x)
    xS = xeig.astype(complex) ** subgraph_size
    mk = min(L0, neighbours_number + subgraph_size)
    binomial_coeff = 1.0

    for k in range(subgraph_size, mk):
        primes[k - 1] += ((-1) ** k / k) * binomial_coeff * ((-1) ** subgraph_size) * xS.sum()
        xS = xS * xeig
        binomial_coeff = binomial_coeff * (subgraph_size - k + neighbours_number) / (1 - subgraph_size + k)

    primes[mk - 1] += ((-1) ** mk / mk) * binomial_coeff * ((-1) ** subgraph_size) * xS.sum()
    return primes


def _matrix_cleaning(A):
    """Remove sources (no incoming edge) then sinks (no outgoing edge)."""
    no_incoming = ~A.any(axis=0)
    A = A[~no_incoming][:, ~no_incoming]
    no_outgoing = ~A.any(axis=1)
    A = A[~no_outgoing][:, ~no_outgoing]
    return A


def _clean_matrix(A):
    """Recursively remove isolated vertices, sinks, and sources until
    the matrix is invariant under such removal -- i.e. every remaining
    vertex sustains at least one cycle."""
    cleaned = A
    while cleaned.shape[0] != _matrix_cleaning(cleaned).shape[0]:
        cleaned = _matrix_cleaning(cleaned)
    return cleaned
