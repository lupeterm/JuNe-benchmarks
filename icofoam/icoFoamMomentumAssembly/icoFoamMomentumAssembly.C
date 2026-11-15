/*---------------------------------------------------------------------------*\
    icoFoamMomentumAssembly

    OpenFOAM baseline of the JuNe/NeoN integration benchmark: assembles the momentum equation
    fvm::ddt(U) + fvm::div(phi, U) - fvm::laplacian(nu, U) of icoFoam RUNS times (option -runs,
    default 5). Every assembly constructs a new fvVectorMatrix, the counterpart of the reset plus
    assembly timed for NeoN and JuNe, between two barriers; every rank prints
        Momentum <run> <seconds>
    and the run's time is that of the slowest rank. Run 0 includes first-use costs.
\*---------------------------------------------------------------------------*/

#include "fvCFD.H"
#include "clockTime.H"
#include <iostream>

int main(int argc, char *argv[])
{
    argList::addNote("Times the assembly of icoFoam's momentum equation (no solve).");
    argList::addOption("runs", "N", "Number of timed assemblies (default 5)");

    #include "setRootCaseLists.H"
    #include "createTime.H"
    #include "createMesh.H"
    #include "createFields.H"

    const label runs = args.getOrDefault<label>("runs", 5);
    Info<< "cells: " << returnReduce(mesh.nCells(), sumOp<label>()) << nl
        << "momentum: ddt(U) + div(phi,U) - laplacian(nu,U), " << runs << " runs" << endl;

    for (label run = 0; run < runs; ++run)
    {
        Pstream::barrier(UPstream::worldComm);
        Foam::clockTime t;
        {
            fvVectorMatrix UEqn
            (
                fvm::ddt(U)
              + fvm::div(phi, U)
              - fvm::laplacian(nu, U)
            );
        }
        Pstream::barrier(UPstream::worldComm);
        const double seconds = t.elapsedTime();
        std::cout << "Momentum " << run << " " << seconds << std::endl;
    }

    return 0;
}
