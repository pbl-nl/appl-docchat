# Configuration
$vaultName = "pbl-openai-a-docchat-kv"
$envFile = ".env"

try {
    # 1. Check if logged into Azure CLI
    Write-Host "Checking Azure CLI session..." -ForegroundColor Cyan
    $account = az account show --query "user" -o tsv 2>$null

    if ($null -eq $account) {
        Write-Host "Error: Not logged in. Please run 'az login' first." -ForegroundColor Red
        exit
    }

    Write-Host "Authenticated as: $account" -ForegroundColor Green

    # 2. Retrieve list of secret names
    Write-Host "Fetching secret list from Key Vault: $vaultName..." -ForegroundColor Cyan
    $secretNames = az keyvault secret list --vault-name $vaultName --query "[].name" -o tsv

    if ([string]::IsNullOrWhiteSpace($secretNames)) {
        Write-Warning "No secrets found or insufficient permissions for Key Vault: $vaultName"
        exit
    }

    # 3. Iterate through secrets and retrieve values
    $envContent = @()
    foreach ($name in $secretNames) {
        Write-Host "Processing: $name" -ForegroundColor Gray
        $value = az keyvault secret show --vault-name $vaultName --name $name --query "value" -o tsv
        $envContent += "$name=$value"
    }

    # 4. Write to .env file
    $envContent | Out-File -FilePath $envFile -Encoding utf8
    Write-Host "`nSuccess! Secrets exported to: $(Get-Item $envFile)" -ForegroundColor Green

}
catch {
    Write-Error "An unexpected error occurred: $_"
}