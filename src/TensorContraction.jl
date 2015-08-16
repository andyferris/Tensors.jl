module TensorContraction

# Part 1: some macros to make input user-friendly
# Part 2: a bunch of new types and generated functions for multiplying tensors
#         (so that determination of permute/etc occur at compile-time only)

##############
### Macros ###
##############
"Macro takes expression and replaces [[ a, ... ]] with [Val{:a},...]"
macro tensor(ex)
    # search through ex and replace all instances of vect with something else
    replace_ref_vect_with_ref_val!(ex)
          
    return esc(ex) # esc() means ex gets returned as-is for running in the caller's scope
end

export @tensor

"Recursive function that replaces [[ a, ... ]] with [Val{:a},...] in an expression tree"
function replace_ref_vect_with_ref_val!(a::Expr)
    for i = 1:length(a.args)
		if isa(a.args[i],Expr)
		    replace_ref_vect_with_ref_val!(a.args[i])
		end
	end
	
    if a.head == :ref
        if isa(a.args[2],Expr) && a.args[2].head == :vect
            # BINGO - we found the double bracket [[ ... ]]
            
            tmp = Vector{Any}(1+length(a.args[2].args))
            tmp[1] = a.args[1]
            for i = 1:length(a.args[2].args)
                tmp2 = parse(":$(a.args[2].args[i])")
                tmp[i+1] = :(Val{$tmp2})
            end
            a.args = tmp
        end
    end
end

###################################################
### These functions generate code for the below ###
###################################################
"Function for generating tensor multiplication code"
function make_multiply_code(Ta,Tb,indices_a,indices_b)
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
    
    T = promote_type(Ta,Tb)
    
    # Build a string of the multiplication code
    sizes_out = "("
    for i = 1:Naa
        if i != 1
            sizes_out = sizes_out * ","
        end
        sizes_out = sizes_out * "size(a.data,$(permute_aa[i]))"
    end
    for i = 1:Nbb
        if i != 1 || Naa > 0
            sizes_out = sizes_out * ","
        end
        sizes_out = sizes_out * "size(b.data,$(permute_bb[i]))"
    end
    sizes_out = sizes_out * ")"
             
    if Nout == 0
        # Inner product
        ex_str = "return (a.data[:].' * permutedims(b.data,$(permute_bc))[:])[1]"
    elseif Nc == 0
        # Outer product
        ex_str = "return Tensor$Nout{$T"
		for i = 1:Naa
		   ex_str = ex_str * ",Val{:$(indices_a[permute_aa[i]])}"
		end
		for i = 1:Nbb
		   ex_str = ex_str * ",Val{:$(indices_b[permute_bb[i]])}"
		end
		ex_str = ex_str * "}(reshape(a.data[:] * b.data[:].',$sizes_out))"
    else
        # Matrix product (or matrix-vector/vector-matrix)
        
        size_aa = "1"
        for i = 1:Naa
            size_aa = size_aa * "*size(a.data,$(permute_aa[i]))"
        end
           
        size_c = "1"
        for i = 1:Nc
            size_c = size_c * "*size(a.data,$(permute_ac[i]))"
        end 
    
        size_bb = "1"
        for i = 1:Nbb
            size_bb = size_bb * "*size(b.data,$(permute_bb[i]))" 
        end  
		   
		ex_str = "return Tensor$Nout{$T"
		for i = 1:Naa
		   ex_str = ex_str * ",Val{:$(indices_a[permute_aa[i]])}"
		end
		for i = 1:Nbb
		   ex_str = ex_str * ",Val{:$(indices_b[permute_bb[i]])}"
		end
		ex_str = ex_str * "}(reshape(reshape(permutedims(a.data,$(vcat(permute_aa,permute_ac))),($size_aa,$size_c)) * reshape(permutedims(b.data,$(vcat(permute_bc,permute_bb))),($size_c,$size_bb)),$sizes_out))" 
    end
    
    #display(parse(ex_str))
    
    return parse(ex_str)
end

"Function for generating tensor creation code"
function make_build_tensor_code(T,indices)
    N = length(indices)
    Nc = 0
    Na = 0

    pairs = zeros(Int64,(2,N))
    singles = zeros(Int64,N)
    indexstr = "["
    for i = 1:N
        m = find(x->x==indices[i],indices[i+1:N]) + i
        Nm = length(m)
        display(pairs)
        if Nm > 1
            error("Index $(indices[i]) appears more than twice (pairwise contractions only)")
        elseif Nm == 1
            Nc += 1
            pairs[1,Nc] = i 
            pairs[2,Nc] = m[1]
            indexstr = indexstr * "i$(Nc)"
        elseif in(i,pairs[2,:])
            indexstr = indexstr * "i$(find(x->x==i,pairs[2,:])[1])"
        else
            Na += 1
            singles[Na] = i
            indexstr = indexstr * ":"
        end
        if i < N
            indexstr = indexstr * ","
        end
    end
    indexstr = indexstr * "]"
    
    ex_str = ""
    
    if Nc == 0
        ex_str = "Tensor$N{$T,"
        for i = 1:N
            ex_str = ex_str * ":$(indices[i])"
            if i < N
                ex_str = ex_str * ","
            end
        end
        ex_str = ex_str * "}(data)"
    else
        # Perform a partial trace...
        ex_str = "begin; data_tmp = zeros(T,"
        for i = 1:Na
             ex_str = ex_str * "size(data,$(singles[i]))"
             if i < Na
                 ex_str = ex_str * ","
             end
        end
        ex_str = ex_str * "); "
        
        for i = 1:Nc
            ex_str = ex_str * "for i$i = 1:size(data,$(pairs[1,i])); "
        end
        ex_str = ex_str * "data_tmp += data$indexstr; "
        for i = 1:Nc
            ex_str = ex_str * "end; "
        end 
        
        # Assign data to an array
        ex_str = ex_str * "Tensor$Na{$T,"
		for i = 1:Na
			ex_str = ex_str * "Val{:$(indices[singles[i]])}"
			if i < Na
				ex_str = ex_str * ","
			end
        end
        ex_str = ex_str * "}(data_tmp); end"
    end

    return parse(ex_str)
end

"Function for generating tensor getindex code"
function make_gettensorindex_code(T,indices_in,indices_out)
    N = length(indices_in)
    permutation = zeros(Int64,N)
    for i = 1:N
        m = find(x->x==indices_in[i],indices_out)
        if length(m) != 1
            error("Output indices $indices_out must match up with tensor indices $indices_in")
        end
        permutation[i] = m[1]
    end
    return :(permutedims(tensor.data,$permutation))
end

"Function for generating tensor setindex code"
function make_settensorindex_code(T,indices_in,indices_out)
    N = length(indices_in)
    permutation = zeros(Int64,N)
    for i = 1:N
        m = find(x->x==indices_in[i],indices_out)
        if length(m) != 1
            error("Output indices $indices_out must match up with tensor indices $indices_in")
        end
        permutation[i] = m[1]
    end
    return :(dataout = permutedims(tensor.data,$permutation))
end

macro maketensor(n::Int64)
    template_str = "index1"
    template_outstr = "out1"
    val_str = "Val{index1}"
    type_str = "::Type{Val{index1}}"
    out_str = "::Type{Val{out1}}"
    for i = 2:n
        template_str = template_str * ",index$i"
        template_outstr = template_outstr * ",out$i"
        val_str = val_str * ",Val{index$i}"
        type_str = type_str * ",::Type{Val{index$i}}"
        out_str = out_str * ",::Type{Val{out$i}}"
    end
    
    ex_str = """
		type Tensor$n{T,$template_str}
			data::Array{T,$n}
		end

		@generated function getindex{T,$template_str}(data::Array{T,$n},$type_str)
			# Need to allow for the possibility of repeated indices, in which case we perform 
			# a (partial) trace
			indices = [$template_str]
	
			return make_build_tensor_code(T,indices)    
		end

		@generated function getindex{T,$template_str,$template_outstr}(tensor::Tensor$n{T,$val_str},$out_str)
			indices_in = [$template_str]
			indices_out = [$template_outstr]
	
			return make_gettensorindex_code(T,indices_in,indices_out)
		end

		@generated function setindex!{T,$template_str,$template_outstr}(dataout::Array{T,$n},tensor::Tensor$n{T,$val_str},$out_str)
			indices_in = [$template_str]
			indices_out = [$template_outstr]
	
			return make_settensorindex_code(T,indices_in,indices_out)
		end
	"""
	
	a_strs = Vector{String}(n)
	b_strs = Vector{String}(n)
	va_strs = Vector{String}(n)
	vb_strs = Vector{String}(n)
	
	a_strs[1] = "index1a"
	b_strs[1] = "index1b"
	va_strs[1] = "Val{index1a}"
	vb_strs[1] = "Val{index1b}"
	
	for i = 2:n
	    a_strs[i] = a_strs[i-1] * ",index$(i)a"
	    b_strs[i] = b_strs[i-1] * ",index$(i)b"
	    va_strs[i] = va_strs[i-1] * ",Val{index$(i)a}"
	    vb_strs[i] = vb_strs[i-1] * ",Val{index$(i)b}"
	end
	
	for i = 1:n
	    ex_str = ex_str * """
	        @generated function *{Ta,Tb,$(a_strs[i]),$(b_strs[n])}(a::Tensor$i{Ta,$(va_strs[i])},b::Tensor$n{Tb,$(vb_strs[n])})
				indices_a = [$(a_strs[i])]
				indices_b = [$(b_strs[n])]
	
				return make_multiply_code(Ta,Tb,indices_a,indices_b)
			end
		"""
	end
	
	for i = n-1:-1:1
	    ex_str = ex_str * """
	        @generated function *{Ta,Tb,$(a_strs[n]),$(b_strs[i])}(a::Tensor$n{Ta,$(va_strs[n])},b::Tensor$i{Tb,$(vb_strs[i])})
				indices_a = [$(a_strs[n])]
				indices_b = [$(b_strs[i])]
	
				return make_multiply_code(Ta,Tb,indices_a,indices_b)
			end
		"""
	end

    if n == 4 
        println("begin; "*ex_str*"end")
    end

    return(esc(parse("begin; "*ex_str*"end")))
end

# We use the type-system to fully determine each permute and copy

# Since Julia doesn't have "generated types" or type-families, the only straightforward
# way of doing things is explicitly, defining a type for each dimension of tensor.

# Presumably, one could write a macro to make all this??

import Base.*
import Base.getindex
import Base.setindex!
import Base.display

######################
### Rank-1 Tensors ###
######################
type Tensor1{T,index1}
    data::Array{T,1}
end

getindex{T,index1}(data::Array{T,1},::Type{Val{index1}}) = Tensor1{T,Val{index1}}(data)

getindex{T,index1}(tensor::Tensor1{T,Val{index1}},::Type{Val{index1}}) = tensor.data

function setindex!{T,index1}(dataout::Array{T,1},tensor::Tensor1{T,Val{index1}},key::Type{Val{index1}})
    dataout[:] = tensor.data[:]
end

@generated function *{Ta,Tb,index1a,index1b}(a::Tensor1{Ta,Val{index1a}},b::Tensor1{Tb,Val{index1b}})
    T = promote_type(Ta,Tb)
    if index1a == index1b # inner product
        return :((a.data.' * b.data)[1])
    else # outer product
        return :(Tensor2{$T,Val{index1a},Val{index1b}}(a.data*b.data.'))
    end
end

function display{T,index1}(a::Tensor1{T,Val{index1}})
    println("One dimensional tensor of type $T with index $index1 and data:")
    show(a.data)
end
     
######################
### Rank-2 Tensors ###
######################
type Tensor2{T,index1,index2}
    data::Array{T,2}
end

getindex{T,index1}(data::Array{T,2},::Type{Val{index1}},::Type{Val{index1}}) = trace(data)
getindex{T,index1,index2}(data::Array{T,2},::Type{Val{index1}},::Type{Val{index2}}) = Tensor2{T,Val{index1},Val{index2}}(data)

@generated function getindex{T,index1,index2,out1,out2}(tensor::Tensor2{T,Val{index1},Val{index2}},::Type{Val{out1}},::Type{Val{out2}})
    if out1 == index1 && out2 == index2 # the array
        return :(tensor.data)
    elseif out1 == index2 && out2 == index1 # the transpose of the array
        return :(tensor.data.')
    else
        error("Indices do not match")
    end
end

@generated function setindex!{T,index1,index2,out1,out2}(dataout::Array{T,3},tensor::Tensor2{T,Val{index1},Val{index2}},::Type{Val{out1}},::Type{Val{out2}})
    if out1 == index1 && out2 == index2
        return :(outdata = tensor.data)
    elseif out1 == index2 && out2 == index1
        return :(outdata = tensor.data.')
    else
        error("Indices do not match")
    end
end

@generated function *{Ta,Tb,index1a,index2a,index1b}(a::Tensor2{Ta,Val{index1a},Val{index2a}},b::Tensor1{Tb,Val{index1b}})
    indices_a = [index1a, index2a]
    indices_b = [index1b]
    
    return make_multiply_code(Ta,Tb,indices_a,indices_b)
end

@generated function *{Ta,Tb,index1a,index1b,index2b}(a::Tensor1{Ta,Val{index1a}},b::Tensor2{Tb,Val{index1b},Val{index2b}})
    indices_a = [index1a]
    indices_b = [index1b,index2b]
    
    return make_multiply_code(Ta,Tb,indices_a,indices_b)
end

@generated function *{Ta,Tb,index1a,index2a,index1b,index2b}(a::Tensor2{Ta,Val{index1a},Val{index2a}},b::Tensor2{Tb,Val{index1b},Val{index2b}})
    indices_a = [index1a, index2a]
    indices_b = [index1b, index2b]
    
    return make_multiply_code(Ta,Tb,indices_a,indices_b)
end

######################
### Rank-3 Tensors ###
######################
type Tensor3{T,index1,index2,index3}
    data::Array{T,3}
end

@generated function getindex{T,index1,index2,index3}(data::Array{T,3},::Type{Val{index1}},::Type{Val{index2}},::Type{Val{index3}})
    # Need to allow for the possibility of repeated indices, in which case we perform 
    # a (partial) trace
    indices = [index1,index2,index3]
    
    return make_build_tensor_code(T,indices)    
end

@generated function getindex{T,index1,index2,index3,out1,out2,out3}(tensor::Tensor3{T,Val{index1},Val{index2},Val{index3}},::Type{Val{out1}},::Type{Val{out2}},::Type{Val{out3}})
    indices_in = [index1,index2,index3]
    indices_out = [out1,out2,out3]
    
    return make_gettensorindex_code(T,indices_in,indices_out)
end

@generated function setindex!{T,index1,index2,index3,out1,out2,out3}(dataout::Array{T,3},tensor::Tensor3{T,Val{index1},Val{index2},Val{index3}},::Type{Val{out1}},::Type{Val{out2}},::Type{Val{out3}})
    indices_in = [index1,index2,index3]
    indices_out = [out1,out2,out3]
    
    return make_settensorindex_code(T,indices_in,indices_out)
end

@generated function *{Ta,Tb,index1a,index2a,index3a,index1b}(a::Tensor3{Ta,Val{index1a},Val{index2a},Val{index3a}},b::Tensor1{Tb,Val{index1b}})
    indices_a = [index1a, index2a, index3a]
    indices_b = [index1b]
    
    return make_multiply_code(Ta,Tb,indices_a,indices_b)
end

@generated function *{Ta,Tb,index1a,index2a,index3a,index1b,index2b}(a::Tensor3{Ta,Val{index1a},Val{index2a},Val{index3a}},b::Tensor2{Tb,Val{index1b},Val{index2b}})
    indices_a = [index1a, index2a, index3a]
    indices_b = [index1b, index2b]
    
    return make_multiply_code(Ta,Tb,indices_a,indices_b)
end

@generated function *{Ta,Tb,index1a,index2a,index3a,index1b,index2b,index3b}(a::Tensor3{Ta,Val{index1a},Val{index2a},Val{index3a}},b::Tensor3{Tb,Val{index1b},Val{index2b},Val{index3b}})
    indices_a = [index1a, index2a, index3a]
    indices_b = [index1b, index2b, index3b]
    
    return make_multiply_code(Ta,Tb,indices_a,indices_b)
end

@generated function *{Ta,Tb,index1a,index2a,index1b,index2b,index3b}(a::Tensor2{Ta,Val{index1a},Val{index2a}},b::Tensor3{Tb,Val{index1b},Val{index2b},Val{index3b}})
    indices_a = [index1a, index2a]
    indices_b = [index1b, index2b, index3b]
    
    return make_multiply_code(Ta,Tb,indices_a,indices_b)
end

@generated function *{Ta,Tb,index1a,index1b,index2b,index3b}(a::Tensor2{Ta,Val{index1a}},b::Tensor3{Tb,Val{index1b},Val{index2b},Val{index3b}})
    indices_a = [index1a]
    indices_b = [index1b, index2b, index3b]
    
    return make_multiply_code(Ta,Tb,indices_a,indices_b)
end

######################
### Rank>4 Tensors ###
######################
# The pattern emerges above - can easily generate with a macro
@maketensor 4
@maketensor 5
@maketensor 6
@maketensor 7
@maketensor 8
@maketensor 9
@maketensor 10
@maketensor 11
@maketensor 12
@maketensor 13
@maketensor 14
@maketensor 15
@maketensor 16
@maketensor 17
@maketensor 18
@maketensor 19
@maketensor 20
@maketensor 21
@maketensor 22
@maketensor 23
@maketensor 24
@maketensor 25
@maketensor 26
@maketensor 27
@maketensor 28
@maketensor 29
@maketensor 30
@maketensor 31
@maketensor 32

# Would someone want larger tensors?? Can we make it user selectable - say 16 by default?
    
end # module
