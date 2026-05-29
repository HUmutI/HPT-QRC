push!(LOAD_PATH,"../package/QuantumCircuits_demo/src","../package/VQC_demo_cuda/src")
using VQC, VQC.Utilities
using QuantumCircuits, QuantumCircuits.Gates
using Flux:train!
using Flux
using Random
using Statistics
using StatsBase
using LinearAlgebra
using CUDA
import LinearAlgebra: tr

struct MyModel
    L::Int
    OutLen::Int
    W::Matrix{Float64}
    Features::Vector{String}
end
MyModel(L::Int,OutLen::Int,Features::Vector{String},) = MyModel(L,OutLen,zeros(L,OutLen),Features)

function tr(m::CuDensityMatrixBatch)
    mat = storage(m)
    remat = reshape(mat,2^m.nqubits,2^m.nqubits,m.nitems)
    x=zeros(eltype(m),m.nitems)
    for i in 1:m.nitems
        x[i]=CUDA.tr(remat[:,:,i])
    end
    return x[1]
end


# (max(RV_raw...)-min(RV_raw...))^2
Max_RV=-1.2543188032019446
Min_RV=-4.7722718186046515
coe = (Max_RV-Min_RV)^2
dif = (Max_RV-Min_RV)

function denormalization(x1,x2,y,a,b)
    xmax=x1
    xmin=x2
    f(z)=(z-a)*(xmax-xmin)/(b-a)+xmin
    return map(f,y)
end

function ⊗(A::DensityMatrix,B::DensityMatrix)
    return DensityMatrix(kron(storage(A),storage(B)),nqubits(A)+nqubits(B))
end

function ⊗(A::CuDensityMatrixBatch,B::CuDensityMatrixBatch)
    return CuDensityMatrixBatch(kron(storage(A),storage(B)),A.nqubits+B.nqubits,1)
end

import Base: *
function *(A::Union{Matrix,CuArray}, B::DensityMatrix)
    return DensityMatrix(Array(A*CuArray(storage(B))))
end
function *(B::DensityMatrix,A::Union{CuArray,Adjoint})
    return DensityMatrix(Array(CuArray(storage(B))*A))
end

function *(A::Union{Matrix,CuArray}, B::CuDensityMatrixBatch)
    return CuDensityMatrixBatch(A*storage(B),B.nqubits,B.nitems)
end
function *(B::CuDensityMatrixBatch,A::Union{CuArray,Adjoint})
    return CuDensityMatrixBatch(storage(B)*A,B.nqubits,B.nitems)
end

function Qreservoir(nqubit,ps)
    H=QubitsOperator()
    for i in 1:nqubit
        for j in i+1:nqubit
            H+=QubitsTerm(i=>"X",j=>"X",coeff=ps[i,j])
        end
    end
    
    for i in 1:nqubit
        H+=QubitsTerm(i=>"Z",coeff=1)
    end
    return H
end

function normalization(x,a,b)
    xmax=maximum(x)
    xmin=minimum(x)
    f(z)=(b-a)*(z-xmin)/(xmax-xmin)+a
    return map(f,x)
end

function denormalization(x,y,a,b)
    xmax=maximum(x)
    xmin=minimum(x)
    f(z)=(z-a)*(xmax-xmin)/(b-a)+xmin
    return map(f,y)
end


MAPE(x,y) = mean(abs.((x-y).*dif./((y.+1)*dif.+Min_RV)))*100
MAPE_std(x,y) = std(abs.((x-y).*dif./((y.+1)*dif.+Min_RV)))*100

MAE(x,y) = mean(abs.((x-y).*dif))
MAE_std(x,y) = std(abs.((x-y).*dif))

MSE(x,y) = mean(((x-y).^2).*coe)
MSE_std(x,y) = std(((x-y).^2).*coe)

RMSE(x,y) = sqrt(mean(((x-y).^2).*coe))



function Quantum_Reservoir(Data, features, QR, Observable, K_delay, VirtualNode, τ, nqubit)
    N = length(Observable)
    L = size(Data,1)
    
    InputSize = length(features)

    Output = zeros(N*VirtualNode,L)
    
    δτ = τ/VirtualNode

    δU = CuArray(exp(-im*δτ*Matrix(matrix(QR))))
    U = CuArray(exp(-im*τ*Matrix(matrix(QR))))

    for l in (K_delay+1):L
        cir = QCircuit()
        for i in 1:InputSize
            push!(cir,RyGate(i,rand(),isparas=true))
        end
        ρᵣ = CuDensityMatrixBatch{ComplexF32}(nqubit-InputSize,1)
        # if InputSize == nqubit
        #     ρᵣ=1
        # end
        #ρ = DensityMatrix(nqubit)
        for k in K_delay:-1:1

            ρ₁ = CuDensityMatrixBatch{ComplexF32}(InputSize,1)
            para = [Data[l-k,str] for str in features]
            cir(para.*pi)
            if (nqubit-InputSize)==0
                ρ = cir(ρ₁)
            else
                ρ = ρᵣ⊗(cir(ρ₁))
            end
            if k!=1
                ρ = U*ρ*U'
                ρᵣ=partial_tr(ρ, Vector(1:InputSize))
            else
                it=1
                for v in 1:VirtualNode
                    ρ = δU*ρ*δU'
                    for n in 1:N 
                        Output[it,l] = vec(real(expectation(B[n],ρ)))[1]
                        it+=1
                    end
                end
            end
            #ρᵣ=partial_tr(ρ, Vector(1:InputSize))
        end
    end
    return Output
end


function compute_qlike(forecasts,  actuals)
    """
    Compute the QLIKE (Quasi-Likelihood) loss function for evaluating forecasting accuracy.
    forecasts: Forecasted variance (sigma squared from a model)
    actuals: Realized variance (actual observed variance)
    """
    # Using absolute values of forecasts and actuals
    forecasts =abs.((forecasts.+1)*dif.+Min_RV)
    actuals = abs.((actuals.+1)*dif.+Min_RV)

    # Calculate the ratio and ensure it's positive
    ratio = actuals ./ forecasts

    # Compute QLIKE
    qlike = sum(ratio - log.(ratio).-1)
    return qlike
end

function compute_qlike2(forecasts, actuals)
    # Using absolute values of forecasts and actuals
    forecasts =(forecasts.+1)*dif.+Min_RV
    actuals = (actuals.+1)*dif.+Min_RV

    # Calculate the ratio and ensure it's positive
    ratio = exp.(actuals) ./ exp.(forecasts)

    # Compute QLIKE
    qlike = sum(ratio -(actuals-forecasts).-1)
    return qlike
end

function coeff_matrix(N,J)
    m=rand(N,N)
    m=(m+transpose(m))./2
    for i in 1:N
        m[i,i]=0.0
    end
    return m./max(eigvals(m)...).*J
end

function wave(y)
    L=length(y)
    w=zeros(L-1)
    for i in 1:L-1
        w[i]=sign(y[i+1]-y[i])
    end
    return w
end
function hitrate(x,y)
    L=length(x)
    x=vcat(-0.5704088242386152,x)
    y=vcat(-0.5704088242386152,y)
    wx=wave(x)
    wy=wave(y)
    return sum(wx.==wy)/L
end


function (c::QCircuit)(p::Vector)
    return reset_parameters!(c,p)
end

function (c::QCircuit)(ρ::DensityMatrix)
    return c*ρ
end

function (c::QCircuit)(ρ::CuDensityMatrixBatch)
    return c*ρ
end
# println("Virtual node: 0")
# println("Train_interval:$(wi)")
# println("Hit rate:$(hitrate(P1,y_test))")
# println("MAPE: $(MAPE(P1,y_test))")
# println("MAE: $(MAE(P1,y_test))")
# println("MSE: $(MSE(P1,y_test))")
# println("RMSE: $(RMSE(P1,y_test))")


# anim = @animate for i in 1:10
#     plot(Vector(K+1:60),[(Ws[i]*signals[i])'[K+1:end],RV[(K+1):60]],label=["Predict" "Target"],lw=3,legendfontsize=18,titlefontsize=24,guidefontsize=24,tickfontsize=18,bottom_margin=5mm,left_margin = 5mm,size=(1000,600),ylims=(-1,-0.3),legend=:bottomleft,xlabel="Time", ylabel="Rv")
#     quiver!([20,30], [-0.45,-0.45], quiver=([-20,20], [0,0]),lw=3)
#     quiver!([56,62], [-0.45,-0.45], quiver=([-4,5], [0,0]),lw=3)
#     vline!([48],lw=2, label= false)
#     annotate!(25,-0.45,("Train",24))
#     annotate!(59,-0.45,("Test",24))
#     annotate!(30,-0.95,("Virtual node=$i",24))
# end
# i=10
# plot(Vector(K+1:Ti+12),[(Ws[i]*signals[i])'[K+1:end],vcat(y_train,y_test)],label=["Predict" "Target"],lw=3,legendfontsize=18,titlefontsize=24,guidefontsize=24,tickfontsize=18,bottom_margin=5mm,left_margin = 5mm,size=(1000,600),ylims=(-1,-0.3),legend=:bottomleft,xlabel="Time", ylabel="Rv")
# quiver!([20,30], [-0.45,-0.45], quiver=([-20,20], [0,0]),lw=3)
# quiver!([56,62], [-0.45,-0.45], quiver=([-4,5], [0,0]),lw=3)
# vline!([51],lw=2, label= false)
# annotate!(25,-0.45,("Train",24))
# annotate!(59,-0.45,("Test",24))
# annotate!(30,-0.95,("Virtual node=$i",24))

function shift(V::Vector{T},step::Int) where T
    V1=zeros(T, length(V))
    V1[step+1:end] = V[1:end-step]
    return V1
end

function shift(V::Matrix{T},step::Int) where T
    V1=zeros(T, size(V))
    V1[step+1:end,:] = V[1:end-step,:]
    return V1
end

function rolling(V::Vector{T},window::Int) where T
    M = zeros(T,length(V),window)
    for i in 1:window
        M[i:end,i]=V[1:end-i+1]
    end
    return M
end

function Quantum_Reservoir_util(Data, features, QR, Observable, K_delay, VirtualNode, τ, nqubit)
    N = length(Observable)
    L = size(Data,1)
    
    InputSize = length(features)

    Output = zeros(N*VirtualNode,L)
    
    δτ = τ/VirtualNode

    δU = CuArray(exp(-im*δτ*Matrix(matrix(QR))))
    U = CuArray(exp(-im*τ*Matrix(matrix(QR))))

    for l in (K_delay+1):L
        cir = QCircuit()
        for i in 1:InputSize
            push!(cir,RyGate(i,rand(),isparas=true))
        end
        ρᵣ = CuDensityMatrixBatch{ComplexF32}(nqubit-InputSize,1)
        for k in K_delay:-1:1

            ρ₁ = CuDensityMatrixBatch{ComplexF32}(InputSize,1)
            para = Vector(Data[l-1,(k-1)*InputSize+1:k*InputSize])#[Data[l-k,str] for str in features]
            cir(para.*pi)
            ρ = ρᵣ⊗(cir(ρ₁))
            if k!=1
                ρ = U*ρ*U'
                ρᵣ=partial_tr(ρ, Vector(1:InputSize))
            else
                it=1
                for v in 1:VirtualNode
                    ρ = δU*ρ*δU'
                    for n in 1:N 
                        Output[it,l] = vec(real(expectation(B[n],ρ)))[1]
                        it+=1
                    end
                end
            end
        end
    end
    return Output
end


function Quantum_Reservoir_single(Input, U, U1, Observable, VirtualNode, nqubit,bias)
    N = length(Observable)
    K, InputSize = size(Input)

    cir = QCircuit()
    for i in 1:InputSize
        push!(cir,RyGate(i,rand(),isparas=true))
    end

    cir2 = QCircuit()
    for i in 1:nqubit-InputSize
        push!(cir2,RyGate(i,rand(),isparas=true))
    end
    cir2(bias.*pi)
    ρᵣ = DensityMatrix(nqubit-InputSize)
    ρᵣ=cir2(ρᵣ)
    Output = zeros(N*VirtualNode)

    for k in 1:K
        ρ₁ = DensityMatrix(InputSize)
        cir(Input[k,:].*pi)
        ρ = ρᵣ⊗(cir(ρ₁))
        if k!=K
            ρ = U*ρ*U'
            ρᵣ=partial_tr(ρ, Vector(1:InputSize))
        else
            it=1
            for v in 1:VirtualNode
                ρ = U1*ρ*U1'
                for n in 1:N 
                    Output[it] = real(expectation(Observable[n],ρ))
                    it+=1
                end
            end
        end
        #ρᵣ=partial_tr(ρ, Vector(1:InputSize))
    end
    return Output
end

# function Quantum_Reservoir(Data, features, QR, Observable, K_delay, VirtualNode, τ, nqubit)
#     N = length(Observable)
#     L = size(Data,1)
#     x_data = zeros(N*VirtualNode,L)
#     Input = Matrix(Data[:,features])
#     δv = τ/VirtualNode
#     U = exp(-im*τ*Matrix(matrix(QR)))
#     U1 = exp(-im*δv*Matrix(matrix(QR)))
#     Threads.@threads for l in K_delay+1:L
#         x_data[:,l] = Quantum_Reservoir_single(Input[l-K_delay:l,:], U, U1, Observable, VirtualNode, nqubit)
#     end
#     return x_data

# end