module Tensors

"A multi-dimensional array decorated with index names"
type Tensor{T,N,Indices<:Tuple} <: AbstractArray{T,N}
    data::Array{T,N}
end

export Tensor

# Conversion from Array should perform some basic checks
import Base.convert
@generated function convert{T,N,Indices<:Tuple}(::Type{Tensor{T,N,Indices}}, data::Array{T,N})
    # Test if Indices has correct length
	if (N == length(Indices.parameters))
		# Test if there are repeated indices
		if (N == length(unique(Indices.parameters)))
			return :(Tensor{$T,$N,$Indices}(data))
		else # We could perform a partial trace, but convert() should preserve data as best as possible.
		    return :(error("Indices $(Indices) are not unique"))
		end
	else
		return :(error("Attempted to assign $(length(Indices.parameters)) indices for $N-dimensional tensor"))
	end
end

# Simple conversion back to an array
convert{T,N,Indices<:Tuple}(::Type{Array{T,N}}, tensor::Tensor{T,N,Indices}) = tensor.data

# Inherit Array definitions and behaviours
import Base.getindex
import Base.setindex!
import Base.eltype
import Base.length
import Base.ndims
import Base.size
import Base.eachindex
import Base.stride
import Base.strides
import Base.linearindexing
import Base.*
import Base.+
import Base.-
import Base.conj
import Base.(.*)
import Base.(./)
import Base.(.^)
import Base.exp
import Base.log
import Base.sin
import Base.cos
import Base.tan
import Base.sinh
import Base.cosh
import Base.tanh
import Base.eig
import Base.svd
import Base.eigs
import Base.svds

# Informational...
eltype{T}(::Tensor{T}) = T
length(t::Tensor) = length(t.data)
ndims{T,N}(::Tensor{T,N}) = N
size(t::Tensor) = size(t.data)
size(t::Tensor,n::Integer) = size(t.data,n)
eachindex(t::Tensor) = eachindex(t.data)
stride(t::Tensor,k::Integer) = eachindex(t.data,k)
strides(t::Tensor) = eachindex(t.data)

# Requirements of AbstractArray (could do DenseArray, I suppose?)
linearindexing{T,N,Indices}(::Type{Tensor{T,N,Indices}}) = linearindexing(Array) # == Base.linearfast()
getindex{T,N,Indices}(t::Tensor{T,N,Indices},i::Int) = getindex(t.data,i)
setindex!{T,N,Indices}(t::Tensor{T,N,Indices},v,i::Int) = setindex!(t.data,v,i)
# similar()...

# Constructing a Tensor from an Array. 
@generated function getindex{T,N,Indices<:Tuple}(data::Array{T,N},::Type{Indices})
    idx = svec_to_vec(Indices.parameters)
    
    # Test if Indices has correct length
	if (N != length(idx)); return :(error("Attempted to assign $(length(Indices.parameters)) indices for $N-dimensional tensor")); end
	
	# Test if there are repeated indices
	if (N == length(unique(idx))); return :(Tensor{$T,$N,$Indices}(data)); end
	
	# Perform a (full or partial) trace
	uniques = unique(idx)
    ex_str = "begin; data_tmp = zeros($T,("
	n_sums = 0
	unique_idx = Any[]
	idx_strs = Vector{String}(N)
	for i = 1:length(uniques)
	    x = find(x->x==uniques[i],idx)
	    if length(x) == 1
	        idx_strs[x] = ":"
	        push!(unique_idx,uniques[i])
	        if length(unique_idx) > 1; ex_str = ex_str * ","; end
	        ex_str = ex_str * "size(data,$(x[1]))"
	    end
	end
	ex_str = ex_str * ")); "
	
	if length(unique_idx) == 0; ex_str = "begin; data_tmp = zero($T); "; end
		
	for i = 1:length(uniques)
	    x = find(x->x==uniques[i],idx)
	    if length(x) > 1
	        n_sums += 1
	        idx_strs[x] = "i$(n_sums)"
	        ex_str = ex_str * "for i$(n_sums) = 1:length(size(data,$(x[1]))); "
	    end
	end
	
    ex_str = ex_str * "data_tmp" 
    if length(unique_idx) > 0; ex_str = ex_str * "[:]"; end
    ex_str = ex_str * " += data["
    for i = 1:N
        if i > 1; ex_str = ex_str * ","; end
        ex_str = ex_str * "$(idx_strs[i])"
    end
    ex_str = ex_str * "]"
    if length(unique_idx) > 0; ex_str = ex_str * "[:]"; end
    ex_str = ex_str * "; "
	
	for i = 1:n_sums
	    ex_str = ex_str * "end; "
	end
	
	if length(unique_idx) > 0
		ex_str = ex_str * "Tensor{$T,$(length(unique_idx)),Tuple{"
		for i = 1:length(unique_idx)
			if i > 1; ex_str = ex_str * ","; end
			ex_str = ex_str * "$(unique_idx[i])"
		end
		ex_str = ex_str * "}}(data_tmp); end"
	else
	    ex_str = ex_str * "data_tmp; end"
	end
		
	return parse("$ex_str")
end


# Return an Array from a Tensor, permuting indices to put in request ordering
@generated function getindex{T,N,Indices<:Tuple,Out<:Tuple}(tensor::Tensor{T,N,Indices},::Type{Out})
    if  Out == Indices # no permutation necessary
        return :(tensor.data)
    else # Find the permutation
        idx = Indices.parameters
        out = Out.parameters
        if length(out) != N; return :(error("Attempting to de-index a rank-$N tensor with $(length(out)) indices")); end
        if length(out) != length(unique(out)); return :(error("Indices $Out are not unique")); end
        if length(idx) != N || length(idx) != length(unique(idx)); return :(error("Rank-$N tensor has malformed indices $Indices")); end
        
        return :(permutedims(tensor.data,$(permutation(idx,out))))       
    end
end

# Sets an Array given a tensor and order
function setindex!{T,N,Indices<:Tuple,Out<:Tuple}(arrayout::Array{T,N},tensor::Tensor{T,N,Indices},::Type{Out})
    arrayout = tensor[Out]
end

# Multiplication by a scalar
*{Tx<:Number,T,N,Indices}(x::Tx,tensor::Tensor{T,N,Indices}) = Tensor{promote_type(Tx,T),N,Indices}(x*tensor.data)
*{Tx<:Number,T,N,Indices}(tensor::Tensor{T,N,Indices},x::Tx) = Tensor{promote_type(T,Tx),N,Indices}(tensor.data*x)

# Element-wise addition
@generated function +{T1,N,Indices1,T2,Indices2}(tensor1::Tensor{T1,N,Indices1},tensor2::Tensor{T2,N,Indices2})
    if Indices1 == Indices2
        return :(Tensor{$(promote_type(T1,T2)),N,Indices1}(tensor1.data + tensor2.data))
    else # permute tensor2 first
        idx1 = svec_to_vec(Indices1.parameters)
        idx2 = svec_to_vec(Indices2.parameters)
        return :(Tensor{$(promote_type(T1,T2)),N,Indices1}(tensor1.data + permutedims(tensor2.data,$(permutation(idx1,idx2)))))
    end
end

# Subtraction
-{T,N,Indices}(tensor::Tensor{T,N,Indices}) = Tensor{T,N,Indices}(-tensor.data)
@generated function -{T1,N,Indices1,T2,Indices2}(tensor1::Tensor{T1,N,Indices1},tensor2::Tensor{T2,N,Indices2})
    if Indices1 == Indices2
        return :(Tensor{$(promote_type(T1,T2)),N,Indices1}(tensor1.data - tensor2.data))
    else # permute tensor2 first
        idx1 = svec_to_vec(Indices1.parameters)
        idx2 = svec_to_vec(Indices2.parameters)
        return :(Tensor{$(promote_type(T1,T2)),N,Indices1}(tensor1.data - permutedims(tensor2.data,$(permutation(idx1,idx2)))))
    end
end

# Element-wise multiplication
@generated function .*{T1,N,Indices1,T2,Indices2}(tensor1::Tensor{T1,N,Indices1},tensor2::Tensor{T2,N,Indices2})
    if Indices1 == Indices2
        return :(Tensor{$(promote_type(T1,T2)),N,Indices1}(tensor1.data .* tensor2.data))
    else # permute tensor2 first
        idx1 = svec_to_vec(Indices1.parameters)
        idx2 = svec_to_vec(Indices2.parameters)
        return :(Tensor{$(promote_type(T1,T2)),N,Indices1}(tensor1.data .* permutedims(tensor2.data,$(permutation(idx1,idx2)))))
    end
end

# Element-wise division
@generated function ./{T1,N,Indices1,T2,Indices2}(tensor1::Tensor{T1,N,Indices1},tensor2::Tensor{T2,N,Indices2})
    if Indices1 == Indices2
        return :(Tensor{$(promote_type(T1,T2)),N,Indices1}(tensor1.data ./ tensor2.data))
    else # permute tensor2 first
        idx1 = svec_to_vec(Indices1.parameters)
        idx2 = svec_to_vec(Indices2.parameters)
        return :(Tensor{$(promote_type(T1,T2)),N,Indices1}(tensor1.data ./ permutedims(tensor2.data,$(permutation(idx1,idx2)))))
    end
end

# Element-wise power
.^{Tx<:Number,T,N,Indices}(tensor::Tensor{T,N,Indices},x::Tx) = Tensor{promote_type(T,Tx),N,Indices}(tensor.data .^ x)

# Common element-wise functions such as exp, log, sin, cos, etc
exp{T,N,Indices}(tensor::Tensor{T,N,Indices}) = Tensor{T,N,Indices}(exp(tensor.data))
log{T,N,Indices}(tensor::Tensor{T,N,Indices}) = Tensor{T,N,Indices}(log(tensor.data))
sin{T,N,Indices}(tensor::Tensor{T,N,Indices}) = Tensor{T,N,Indices}(sin(tensor.data))
cos{T,N,Indices}(tensor::Tensor{T,N,Indices}) = Tensor{T,N,Indices}(cos(tensor.data))
tan{T,N,Indices}(tensor::Tensor{T,N,Indices}) = Tensor{T,N,Indices}(tan(tensor.data))
sinh{T,N,Indices}(tensor::Tensor{T,N,Indices}) = Tensor{T,N,Indices}(sinh(tensor.data))
cosh{T,N,Indices}(tensor::Tensor{T,N,Indices}) = Tensor{T,N,Indices}(cosh(tensor.data))
tanh{T,N,Indices}(tensor::Tensor{T,N,Indices}) = Tensor{T,N,Indices}(tanh(tensor.data))
sqrt{T,N,Indices}(tensor::Tensor{T,N,Indices}) = Tensor{T,N,Indices}(sqrt(tensor.data))

# Complex conjugation
conj{T<:Real,N,Indices}(tensor::Tensor{T,N,Indices}) = tensor
conj{T,N,Indices}(tensor::Tensor{T,N,Indices}) = Tensor{T,N,Indices}(conj(tensor.data))


# Tensor multiply that contracts matching indices 
@generated function *{T1,N1,Indices1,T2,N2,Indices2}(tensor1::Tensor{T1,N1,Indices1},tensor2::Tensor{T2,N2,Indices2})
    ex_str = ""
    
    T = promote_type(T1,T2)
    indices_a = svec_to_vec(Indices1.parameters)
    indices_b = svec_to_vec(Indices2.parameters)
    
    
    if length(indices_a) != N1 || length(indices_a) != length(unique(indices_a)); return :(error("Rank-$N tensor has malformed indices $indices_a")); end
    if length(indices_b) != N2 || length(indices_b) != length(unique(indices_b)); return :(error("Rank-$N tensor has malformed indices $indices_b")); end

    ## Prepare the permute order, etc
    #N1 = length(idx1)
    #N2 = length(idx2)
    #Nc = length(intersect(idx1,idx2))
    #(permute1, permute2) = joint_permutation(idx1,idx2)
    
    # If p
   
    # Prepare the permute order, etc
    Na = length(indices_a)
    Nb = length(indices_b)
    Nc = length(intersect(indices_a,indices_b))
    Naa = Na - Nc
    Nbb = Nb - Nc
    Nout = Na + Nb - 2*Nc 
    
    
    permute_aa = zeros(Int64,Na - Nc)
    permute_ac = zeros(Int64,Nc)
    permute_bb = zeros(Int64,Nb - Nc)
    permute_bc = zeros(Int64,Nc)
    
    iaa = 1
    ibb = 1
    iac = 1
    for i = 1:Na
        # TODO: detect more than two indices... (only occurs if user intervenes)
        if in(indices_a[i],indices_b)
            permute_ac[iac] = i
            iac += 1
        else
            permute_aa[iaa] = i
            iaa += 1
        end
    end
    for i = 1:Nb
        if in(indices_b[i],indices_a)
            # Need it to match perfectly to the entry in permute_ac
            entry_a = find(x->x==indices_b[i],indices_a)[1]
            entry_ac = find(x->x==entry_a,permute_ac)[1]
            
            permute_bc[entry_ac] = i
        else
            permute_bb[ibb] = i
            ibb += 1
        end
    end    
    
    
    # Build a string of the multiplication code
    sizes_out = "("
    for i = 1:Naa
        if i != 1
            sizes_out = sizes_out * ","
        end
        sizes_out = sizes_out * "size(tensor1.data,$(permute_aa[i]))"
    end
    for i = 1:Nbb
        if i != 1 || Naa > 0
            sizes_out = sizes_out * ","
        end
        sizes_out = sizes_out * "size(tensor2.data,$(permute_bb[i]))"
    end
    sizes_out = sizes_out * ")"
             
    if Nout == 0
        # Inner product
        ex_str = "return (tensor1.data[:].' * permutedims(tensor2.data,$(permute_bc))[:])[1]"
    elseif Nc == 0
        # Outer product
        ex_str = "return Tensor{$T,$Nout,Tuple{"
		for i = 1:Naa
		    if i > 1; ex_str = ex_str * ","; end
		    ex_str = ex_str * "$(indices_a[permute_aa[i]])"
		end
		for i = 1:Nbb
		    if i > 1 || Naa > 0; ex_str = ex_str * ","; end
		    ex_str = ex_str * "$(indices_b[permute_bb[i]])"
		end
		ex_str = ex_str * "}}(reshape(tensor1.data[:] * tensor2.data[:].',$sizes_out))"
    else
        # Matrix product (or matrix-vector/vector-matrix)
        
        size_aa = "1"
        for i = 1:Naa
            size_aa = size_aa * "*size(tensor2.data,$(permute_aa[i]))"
        end
           
        size_c = "1"
        for i = 1:Nc
            size_c = size_c * "*size(tensor1.data,$(permute_ac[i]))"
        end 
    
        size_bb = "1"
        for i = 1:Nbb
            size_bb = size_bb * "*size(tensor2.data,$(permute_bb[i]))" 
        end  
		   
		ex_str = "return Tensor{$T,$Nout,Tuple{"
		for i = 1:Naa
		    if i > 1; ex_str = ex_str * ","; end
		    ex_str = ex_str * "$(indices_a[permute_aa[i]])"
		end
		for i = 1:Nbb
		    if i > 1 || Naa > 0; ex_str = ex_str * ","; end
		    ex_str = ex_str * "$(indices_b[permute_bb[i]])"
		end
		ex_str = ex_str * "}}(reshape(reshape(permutedims(tensor1.data,$(vcat(permute_aa,permute_ac))),($size_aa,$size_c)) * reshape(permutedims(tensor2.data,$(vcat(permute_bc,permute_bb))),($size_c,$size_bb)),$sizes_out))" 
    end 
        
    return parse(ex_str) 
end

# Eigenvalue with indices arranged
@generated function eig{T,N,Indices,IndicesLeft<:Tuple,IndicesRight<:Tuple,IndicesEig<:Tuple}(tensor::Tensor{T,N,Indices},::Type{IndicesLeft},::Type{IndicesRight},index_eig::Type{IndicesEig} = Tuple{:eig})    
    # We need to permute the matrix into order [indices_left, indices_right]
    idx = svec_to_vec(Indices.parameters)
    idx_left = svec_to_vec(IndicesLeft.parameters)
    idx_right = svec_to_vec(IndicesRight.parameters)
    idx_permuted = vcat(idx_left,idx_right)
    idx_eig = svec_to_vec(IndicesEig.parameters)
    
    if length(idx) != length(idx_permuted); return :(error("Number of indices ($(length(idx_left)), $(length(idx_right))) does not match tensor's $N")); end
    if length(idx_left) == 0; return :(error("At least one left index must be specified")); end
    if length(idx_right) == 0; return :(error("At least one right index must be specified")); end
    if length(idx_eig) != 1; return :(error("Only one index should be associated with the eigenvalues")); end
    
    p = permutation(idx,idx_permuted)
        
    # sizes
    size_left = "size(tensor.data,$(p[1]))"
    resize_left = "size(tensor.data,$(p[1]))"
    for i = 2:length(idx_left)
        size_left = size_left * "*size(tensor.data,$(p[i]))"
        resize_left = resize_left * ",size(tensor.data,$(p[i]))"
    end    
    size_right = "size(tensor.data,$(p[1+length(idx_left)]))"
    #resize_right = "size(tensor.data,$(p[1+length(idx_left)]))"
    for i = 2:length(idx_right)
        size_right = size_right * "*size(tensor.data,$(p[i+length(idx_left)]))"
    #    resize_right = resize_right * ",size(tensor.data,$(p[i+length(idx_left)]))"
    end    
    
    str_left = Vector{String}(length(idx_left))
    for i = 1:length(idx_left); isa(idx_left[i],Symbol) ? str_left[i] = ":$(idx_left[i])" : str_left[i] = "$(idx_left[i])"; end
    #str_right = Vector{String}(length(idx_right))
    #for i = 1:length(idx_right); isa(idx_right[i],Symbol) ? str_right[i] = ":$(idx_right[i])" : str_left[i] = "$(idx_right[i])"; end
    str_eig = (isa(idx_eig[1],Symbol) ? ":$(idx_eig[1])" : "$(idx_eig[1])")
    
    ex_str = "begin; "
    if idx_permuted == idx # No permutation necessary, only reshape.
        ex_str = ex_str * "(D,V) = eig(reshape(tensor.data,($size_left,$size_right))); "
    else # permutation is necessary
        ex_str = ex_str * "(D,V) = eig(reshape(permutedims(tensor.data,$p),($size_left,$size_right))); "
    end  

	ex_str = ex_str * "(Tensor{$T,1,Tuple{$(str_eig)}}(D),Tensor{$T,$(1+length(idx_left)),Tuple{"
	for i = 1:length(idx_left)
		ex_str = ex_str * "$(str_left[i]),"
	end
	ex_str = ex_str * "$(str_eig)}}(reshape(V,($(resize_left),$(size_left))))); "
    
    ex_str = ex_str * "end"
    #display(ex_str)
    return parse(ex_str)
end


@generated function svd{T,N,Indices,IndicesLeft<:Tuple,IndicesRight<:Tuple,IndicesSVD<:Tuple}(tensor::Tensor{T,N,Indices},::Type{IndicesLeft},::Type{IndicesRight},index_svd::Type{IndicesSVD} = Tuple{:svd})    
    # We need to permute the matrix into order [indices_left, indices_right]
    idx = svec_to_vec(Indices.parameters)
    idx_left = svec_to_vec(IndicesLeft.parameters)
    idx_right = svec_to_vec(IndicesRight.parameters)
    idx_permuted = vcat(idx_left,idx_right)
    idx_svd = svec_to_vec(IndicesSVD.parameters)
    
    if length(idx) != length(idx_permuted); return :(error("Number of indices ($(length(idx_left)), $(length(idx_right))) does not match tensor's $N")); end
    if length(idx_left) == 0; return :(begin; n = vecnorm(tensor.data); (one(T),n,tensor/n); end); end
    if length(idx_right) == 0; return :(begin; n = vecnorm(tensor.data); (tensor/n,n,one(T)); end); end
    if length(idx_svd) != 1; return :(error("Only one index should be associated with the singular values")); end
    
    p = permutation(idx,idx_permuted)
        
    # sizes
    size_left = "size(tensor.data,$(p[1]))"
    resize_left = "size(tensor.data,$(p[1]))"
    for i = 2:length(idx_left)
        size_left = size_left * "*size(tensor.data,$(p[i]))"
        resize_left = resize_left * ",size(tensor.data,$(p[i]))"
    end    
    size_right = "size(tensor.data,$(p[1+length(idx_left)]))"
    resize_right = "size(tensor.data,$(p[1+length(idx_left)]))"
    for i = 2:length(idx_right)
        size_right = size_right * "*size(tensor.data,$(p[i+length(idx_left)]))"
        resize_right = resize_right * ",size(tensor.data,$(p[i+length(idx_left)]))"
    end    
    
    str_left = Vector{String}(length(idx_left))
    for i = 1:length(idx_left); isa(idx_left[i],Symbol) ? str_left[i] = ":$(idx_left[i])" : str_left[i] = "$(idx_left[i])"; end
    str_right = Vector{String}(length(idx_right))
    for i = 1:length(idx_right); isa(idx_right[i],Symbol) ? str_right[i] = ":$(idx_right[i])" : str_right[i] = "$(idx_right[i])"; end
    str_svd = (isa(idx_svd[1],Symbol) ? ":$(idx_svd[1])" : "$(idx_svd[1])")
    
    ex_str = "begin; "
    if idx_permuted == idx # No permutation necessary, only reshape.
        ex_str = ex_str * "(U,S,V) = svd(reshape(tensor.data,($size_left,$size_right))); "
    else # permutation is necessary
        ex_str = ex_str * "(U,S,V) = svd(reshape(permutedims(tensor.data,$p),($size_left,$size_right))); "
    end  
    
    ex_str = ex_str * "(Tensor{$T,$(1+length(idx_left)),Tuple{"
	for i = 1:length(idx_left)
		ex_str = ex_str * "$(str_left[i]),"
	end
	ex_str = ex_str * "$(str_svd)}}(reshape(U,($(resize_left),$(size_left)))), Tensor{$T,1,Tuple{$(str_svd)}}(S), Tensor{$T,$(1+length(idx_right)),Tuple{"
	for i = 1:length(idx_right)
		ex_str = ex_str * "$(str_right[i]),"
	end
	ex_str = ex_str * "$(str_svd)}}(reshape(V,($(resize_right),$(size_right))))); "
    
    ex_str = ex_str * "end"
    display(ex_str)
    return parse(ex_str)
end

#################
### Utilities ###
#################

"Convert Simple Vector to Vector"
function svec_to_vec(svec::SimpleVector)
    vec = Vector{Any}(length(svec))
    for i = 1:length(svec)
        vec[i] = svec[i]
    end
    return vec
end

"Function finds the permutation that transforms in into out"
function permutation(in,out)
    N = length(in)
    if length(out) != N; error("Permutation must be between equal-sized sets"); end
    permutation = zeros(Int64,N)
    for i = 1:N
        m = 0
        matches = 0
        for j = 1:N
            if in[i] == out[j]
                matches += 1
                m = j
            end
        end
        if matches != 1
            error("Output $out must match up with input $in")
        end
        permutation[i] = m
    end
    return permutation
end


##############
### Macros ###
##############
"Macro takes expression and replaces [[ a, ... ]] with [Tuple{:a,...}]"
macro tensor(ex)
    # search through ex and replace all instances of vect with something else
    replace_ref_vect_with_tuple!(ex)
          
    return esc(ex) # esc() means ex gets returned as-is for running in the caller's scope
end

export @tensor

#######################
### Code generation ###
#######################
"Recursive function that replaces [[ a, ... ]] with [Tuple{:a,...}] in an expression tree"
function replace_ref_vect_with_tuple!(a::Expr)
    for i = 1:length(a.args)
		if isa(a.args[i],Expr)
		    replace_ref_vect_with_tuple!(a.args[i])
		end
	end
	
    if a.head == :ref
        if isa(a.args[2],Expr) && a.args[2].head == :vect
            # BINGO - we found the double bracket [[ ... ]]
            
            tmp = Vector{Any}(1+length(a.args[2].args))
            tmp[1] = :Tuple
            for i = 1:length(a.args[2].args)
                tmp2 = (a.args[2].args[i])
                if isa(tmp2,Symbol)
                    tmp[i+1] = :($tmp2)
                else
                    tmp[i+1] = tmp2
                end
            end
            a.args[2].head = :curly
            a.args[2].args = tmp
        end
    end
end
   
end # module

