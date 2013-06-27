require 'test/unit'
require 'matrix'
require_relative '../lin_reg/linear_regression'

class LinearRegressionTest < Test::Unit::TestCase

# Give all zeros for x's and some finite values for y's
# Give all zeros for everything
# All negative numbers with each column having a different distribution
# All positive numbers with each column having a different distribution
# Normal dataset - Mix of negative and positive numbers with each column having a different distribution
# All positive numbers with each column having the same distribution but differing in magnitude 
# Large alpha with a normal dataset. 

  def test_x_zeros
    flunk("Test not implemented!")
  end

  def test_all_zeros
    flunk("Test not implemented!")
  end

  def test_all_neg
    flunk("Test not implemented!")
  end

  def test_all_pos
    flunk("Test not implemented!")
  end

  def test_mix
    flunk("Test not implemented!")
  end

  def test_same_dist
    x_mat = Matrix[[2,800], [3,900], [4,1000], [5,1100], [6,1200]]
    y_vec = Vector[290, 340, 390, 440, 490]
    lin_reg = LinearRegression.new(x_mat, y_vec)
    lin_reg.train(0.1)
    assert_in_delta(0.002, lin_reg.cost, 0.1)    
  end
end