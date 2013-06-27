require 'matrix'

class Array
  def avg
    Float(self.reduce(:+))/self.count
  end

  def stdev
    avrg = self.avg
    s_2 = (self.reduce(0.0){|acc, x| acc += Float((x - avrg)**2)})/(self.count - 1)
    Math.sqrt(s_2)
  end

  def normalize
    avrg = self.avg
    sd = self.stdev
    self.map{|x| (x - avrg)/sd}
  end
end

class Vector
  def transpose
    covector
  end

  def Vector.zeros(size)
    Vector.elements(Array.new(size).fill(0))
  end

  def Vector.ones(size)
    Vector.elements(Array.new(size).fill(1))
  end

  def normalize
    Vector.elements(self.to_a.normalize)
  end

  def insert(i, val)
    Vector.elements(self.to_a.insert(i, val))
  end
end

class Matrix
  def normalize_column(i)
    cols = column_vectors
    cols[i] = cols[i].normalize
    Matrix.columns(cols)
  end

  def normalize_columns
    cols = column_vectors
    cols.each_with_index do |col, i|
      cols[i] = col.normalize
    end
    Matrix.columns(cols)
  end

  def insert_column(i, ary)
    cols = column_vectors
    cols.insert(i, ary)
    Matrix.columns(cols)
  end
end