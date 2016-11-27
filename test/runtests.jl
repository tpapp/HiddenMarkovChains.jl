import HiddenMarkovChains
HMC = HiddenMarkovChains
using Distributions
using Base.Test

include("utilities.jl")

######################################################################
# helper functions
######################################################################

@testset "Validating probability vectors and matrices." begin

    @test HMC.isprobvec([0.1, 0.2, 0.7])
    @test !HMC.isprobvec([-0.1, 0.1, 1])
    @test !HMC.isprobvec([0.1, 0.2])
    
    @test HMC.isprobmat(ones(2,3)/3)
    @test !HMC.isprobmat([1.0 2.0;])
    @test !HMC.isprobmat([-1.0 2.0])

end

@testset "Iteration over nonzeros." begin

    @test collect(HMC.column_nz(reshape(1:9, 3, 3), 2))==[(1,4), (2,5), (3,6)]
    @test collect(HMC.column_nz(sparse([1,1,3,2],
                                       [1,2,2,3],
                                       [2,3,5,7]), 2))==[(1,3), (3,5)]
end

######################################################################
# TEST: steady state calculations
######################################################################

@testset "Steady state calculations." begin

    @test_throws DimensionMismatch HMC.steady_state(ones(2,3)/3)
    
    "Steady state calculation error."
    function ss_error(P)
        p = HMC.steady_state(P)
        norm(p'*P-p', 1)
    end
    
    for i in 1:1000
    @test ss_error(random_sparse_stochastic_matrix(9, 4)) < 1e-10
    end
    
    for i in 1:1000
        @test ss_error(random_stochastic_matrix(9)) < 1e-10
    end

end

@testset "Path simulations." begin

    let P = random_sparse_stochastic_matrix(5, 3),
        Q = random_stochastic_matrix(5; ncol=3),
        p = HMC.steady_state(P),
        T = 3, 
        pp = HMC.path_probabilities(P, Q, p, T),
        pp_sim = simulate_path_probabilities(p, P, Q, T, 10000),
        diffs = path_probabilities_difference(pp, pp_sim)
        @test diffs < (0.01,0.01)
        @test length(uncovered_paths(pp, pp_sim)) == 0
    end

end

@testset "Log likelihood." begin

    for i in 1:100
        P = random_stochastic_matrix(3)
        Q = random_stochastic_matrix(3; ncol = 2)
        T = 5
        π = HMC.steady_state(P)
        pp = HMC.path_probabilities(P, Q, π, T)
        
        hi = HMC.HMMinit(π,Q)
        hs = HMC.HMMstep(P,Q)
        
        for i in 1:100
            o = simulate_path(π, P, Q, T)
            log_α = HMC.hmm_init(hi, o[1])
            for i in 2:length(o)
                log_α = HMC.hmm_update(log_α, hs, o[i])
            end
            @test log(pp[o]) ≈ HMC.hmm_log_likelihood(log_α)
        end
    end

end
