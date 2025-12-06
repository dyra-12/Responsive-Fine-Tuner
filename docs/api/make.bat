@ECHO OFF

pushd %~dp0

set SPHINXBUILD=sphinx-build
set SOURCEDIR=.
set BUILDDIR=_build

if "%1"=="" goto html

%SPHINXBUILD% -M %1 %SOURCEDIR% %BUILDDIR% %SPHINXOPTS%
exit /B

:html
%SPHINXBUILD% -M html %SOURCEDIR% %BUILDDIR% %SPHINXOPTS%
