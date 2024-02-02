"""
test_inconvexhull()

Test the inconvexhull function: inconvexhull(training_set::Matrix{Float64}, test_set::Matrix{Float64})
"""
function test_inconvexhull()
    @testset "inconvexhull" begin
        # Create the training set
        training_set = [0. 0; 1 0; 0 1; 1 1]
        
        # Create the test set
        test_set = [0.5 0.5; -0.5 0.5; 0.5 -0.5; 0.0 0.5]
        
        # Test the inconvexhull function
        @test inconvexhull(training_set, test_set, HiGHS.Optimizer) == [true, false, false, true]
    end
end
