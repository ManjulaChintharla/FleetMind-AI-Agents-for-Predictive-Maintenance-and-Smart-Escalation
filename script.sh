#! /usr/bin/sh

echo "Setting up environment..."

az login --use-device-code

# Create random string
guid=$(cat /proc/sys/kernel/random/uuid)
suffix=${guid//[-]/}
suffix=${suffix:0:5}

# Set the necessary variables
RESOURCE_GROUP="rg-contosofleetguard-l${suffix}"
RESOURCE_PROVIDER="Microsoft.MachineLearningServices"
RESOURCE_PROVIDER1="Microsoft.PolicyInsights"
RESOURCE_PROVIDER2="Microsoft.Cdn"
RESOURCE_PROVIDER3="Microsoft.AlertsManagement"
RESOURCE_PROVIDER4="Microsoft.Web"
RESOURCE_PROVIDER5="Microsoft.DBforPostgreSQL"
# REGIONS=("eastus" "westus" "centralus" "northeurope" "westeurope")
REGIONS=("eastus" "westus" "centralus" )
AIREGIONS=("eastus" "southcentralus" )
RANDOM_REGION=${REGIONS[$RANDOM % ${#REGIONS[@]}]}
AI_REGION=${AIREGIONS[$RANDOM % ${#AIREGIONS[@]}]}
WORKSPACE_NAME="amlws-cfg-ws${suffix}"
COMPUTE_INSTANCE="amlicfg-ci${suffix}"
COMPUTE_CLUSTER="amlccfg-aml-cluster"
ADF_NAME="aadataf-cfg-df${suffix}"
Azure_POSTGRESQL_NAME="azpostsql-cfg-psql${suffix}"
USERNAME="citus"
PASSWORD="Fhtest208"
POSTGRESQL_PORT="5432"
DB_NAME="flexibleserverdb"
CONTAINER_NAME="fleetdata"
RULE_NAME="AllowClientIP"
AZURE_OPENAI_NAME="fleetazai${suffix}"
AZURE_SEARCH="fleet-search-service${suffix}"

# Get the subscription ID
SUBSCRIPTION_ID=$(az account show --query id --output tsv)
TENANT_ID=$(az account show --query tenantId --output tsv)

# Automatically retrieve the current user's Object ID
USER_OBJECT_ID=$(az ad signed-in-user show --query id --output tsv)

# Check if the USER_OBJECT_ID is empty or null
if [ -z "$USER_OBJECT_ID" ]; then
    echo "Error: Unable to retrieve the User Object ID. Please check your Azure CLI login or permissions."
    exit 1
else
    echo "User Object ID: $USER_OBJECT_ID"
fi

# Assign roles to the current user
az role assignment create --assignee $USER_OBJECT_ID --role "Key Vault Data Access Administrator" --scope "/subscriptions/$SUBSCRIPTION_ID"
az role assignment create --assignee $USER_OBJECT_ID --role "Key Vault Administrator" --scope "/subscriptions/$SUBSCRIPTION_ID"
az role assignment create --assignee $USER_OBJECT_ID --role "AzureML Compute Operator" --scope "/subscriptions/$SUBSCRIPTION_ID"
az role assignment create --assignee $USER_OBJECT_ID --role "Search Index Data Reader" --scope "/subscriptions/$SUBSCRIPTION_ID"
az role assignment create --assignee $USER_OBJECT_ID --role "Cognitive Services Contributor" --scope "/subscriptions/$SUBSCRIPTION_ID"
az role assignment create --assignee $USER_OBJECT_ID --role "Cognitive Services OpenAI User" --scope "/subscriptions/$SUBSCRIPTION_ID"
az role assignment create --assignee $USER_OBJECT_ID --role "Search Service Contributor" --scope "/subscriptions/$SUBSCRIPTION_ID"
az role assignment create --assignee $USER_OBJECT_ID --role "Azure AI Developer" --scope "/subscriptions/$SUBSCRIPTION_ID"
az role assignment create --assignee $USER_OBJECT_ID --role "Storage Blob Data Contributor" --scope "/subscriptions/$SUBSCRIPTION_ID"


echo "****Roles assigned successfully to User ID: $USER_OBJECT_ID"


# Registring the Azure Machine Learning resource provider in the subscription
echo "Register the Machine Learning resource providers:"
az provider register --namespace $RESOURCE_PROVIDER
az provider register --namespace $RESOURCE_PROVIDER1
az provider register --namespace $RESOURCE_PROVIDER2
az provider register --namespace $RESOURCE_PROVIDER3
az provider register --namespace $RESOURCE_PROVIDER4
az provider register --namespace $RESOURCE_PROVIDER5

# Creating the resource group , workspace and setting to default
echo "Create a resource group and set as default:"
az group create --name $RESOURCE_GROUP --location $RANDOM_REGION
az configure --defaults group=$RESOURCE_GROUP

echo "Creating an Azure Machine Learning workspace:"
az ml workspace create --name $WORKSPACE_NAME 
az configure --defaults workspace=$WORKSPACE_NAME 

# Creatcompute instance
echo "Creating a compute instance with name: " $COMPUTE_INSTANCE
az ml compute create --name ${COMPUTE_INSTANCE} --size Standard_DS3_v2 --type ComputeInstance 

# Create compute cluster
echo "Creating a compute cluster with name: " $COMPUTE_CLUSTER
az ml compute create --name ${COMPUTE_CLUSTER} --size Standard_E4ds_v4 --max-instances 2 --type AmlCompute 

az config set extension.use_dynamic_install=yes_without_prompt

# Create Azure data factory
echo "Creating a Azure data factory with name: " $ADF_NAME
az datafactory create --resource-group $RESOURCE_GROUP --factory-name $ADF_NAME

# Assign the managed identity ID to a variable
ManagedIdentityId=$(az datafactory show --name $ADF_NAME --resource-group $RESOURCE_GROUP --query identity.principalId --output tsv)

# Search for the Key Vault by name
keyVaultName=$(az keyvault list --query "[?contains(name, 'amlwscfgkeyvault')].name | [0]" --output tsv)

# Get the Key Vault Scope
KEY_VAULT_SCOPE="/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.KeyVault/vaults/$keyVaultName"

# Grant "Key Vault Secrets User" role (for secret access)
az role assignment create --assignee $ManagedIdentityId  --role "Key Vault Secrets User"   --scope "$KEY_VAULT_SCOPE"

# Grant "Key Vault Crypto User" role (for key access)
az role assignment create --assignee $ManagedIdentityId   --role "Key Vault Crypto User"     --scope "$KEY_VAULT_SCOPE"

# Grant "Key Vault Certificates Officer" role (for certificate access)
az role assignment create --assignee $ManagedIdentityId  --role "Key Vault Certificates Officer" --scope "$KEY_VAULT_SCOPE"


# Assign RBAC role to managed identity
az role assignment create --assignee $ManagedIdentityId --role "Key Vault Secrets User" --scope "/subscriptions/$(az account show --query id --output tsv)/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.KeyVault/vaults/$keyVaultName"

# Switch Key Vault to Vault Access Policy Mode
az keyvault update --name $keyVaultName  --resource-group $RESOURCE_GROUP --enable-rbac-authorization false

# Create Access Policy for ADF Managed Identity
az keyvault set-policy   --name $keyVaultName  --resource-group $RESOURCE_GROUP   --object-id $ManagedIdentityId   --secret-permissions get list   --key-permissions get list    --certificate-permissions get list

# Search for the storage account by name pattern
storageAccountName=$(az storage account list --query "[?contains(name, 'amlwscfgstorage')].name | [0]" --output tsv)
az role assignment create --assignee $(az account show --query user.name --output tsv) --role "Storage Blob Data Contributor" --scope "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.Storage/storageAccounts/$storageAccountName"





echo "Creating an Azure database for PostgreSQL with name: $Azure_POSTGRESQL_NAME"

az postgres flexible-server create --location westus --resource-group $RESOURCE_GROUP --name $Azure_POSTGRESQL_NAME --admin-user $USERNAME --admin-password $PASSWORD --sku-name Standard_D2s_v3 --tier GeneralPurpose --storage-size 128 --tags "Environment=Dev" --version 14 --high-availability Disabled --public-access All



echo "Username of postgresql is  " : $USERNAME
echo "Password of postgresql is  " : $PASSWORD
echo " Azure postgresql got created " : $Azure_POSTGRESQL_NAME

### Linking Storage account and Postgresql to ADF

# Get Storage Account Key
STORAGE_ACCOUNT_KEY=$(az storage account keys list --resource-group $RESOURCE_GROUP --account-name $storageAccountName --query '[0].value' --output tsv)

# Link Storage Account
echo "🔗 Linking Azure Storage to ADF..."

az datafactory linked-service create \
    --resource-group $RESOURCE_GROUP \
    --factory-name $ADF_NAME \
    --name "BlobStorageLinkedService" \
    --properties '{
        "type": "AzureBlobStorage",
        "typeProperties": {
            "connectionString": "DefaultEndpointsProtocol=https;AccountName='$storageAccountName';AccountKey='$STORAGE_ACCOUNT_KEY';EndpointSuffix=core.windows.net"
        }
    }'

# Link PostgreSQL
echo "🔗 Linking PostgreSQL to ADF..."
az datafactory linked-service create \
    --resource-group $RESOURCE_GROUP \
    --factory-name $ADF_NAME \
    --name "PostgreSQLLinkedService" \
    --properties '{
        "type": "AzurePostgreSql",
        "typeProperties": {
            "connectionString": "Host='$Azure_POSTGRESQL_NAME'.postgres.database.azure.com;Port='$POSTGRESQL_PORT';Database='$DB_NAME';User Id='$USERNAME';Password='$PASSWORD';SslMode=Require;TrustServerCertificate=True;",
            "version": "1.0"
        }
    }'

    
: '
DB_NAME=$(az postgres flexible-server db list \
    --resource-group $RESOURCE_GROUP \
    --server-name $POSTGRES_SERVER_NAME \
    --query "[?name=='$NEW_DB_NAME'].name" --output tsv)
    ------
    #Check if the database was created successfully:
    az postgres flexible-server db list \
    --resource-group $RESOURCE_GROUP \
    --server-name $POSTGRES_SERVER_NAME \
    --query "[].name" --output table
'


# Create a Container in Azure Storage Account

echo "Creating container: $CONTAINER_NAME..."

az storage container create --name $CONTAINER_NAME --account-name $storageAccountName --auth-mode login  # Secure Authentication

#az storage container create --name $CONTAINER_NAME --account-name $storageAccountName

# Assign the Container Name to a Variable
CONTAINER=$(az storage container list \
    --account-name $storageAccountName \
    --query "[?name=='$CONTAINER_NAME'].name" --output tsv)


# Verify and display the container name
if [ "$CONTAINER" == "$CONTAINER_NAME" ]; then
    echo " Storage Container Created: $CONTAINER"
else
    echo "Failed to create the container."
    exit 1
fi
# move fleet_data file to the storage account container.
az storage blob upload --account-name $storageAccountName --container-name $CONTAINER_NAME --name fleet_data.csv --file ../Data/fleet_data_10k.csv --auth-mode key

#Create Azure OpenAI service
az cognitiveservices account create --name $AZURE_OPENAI_NAME --resource-group $RESOURCE_GROUP --location $AI_REGION --kind OpenAI --sku S0  --yes

#Create AzureAI Search service
az search service create  --name $AZURE_SEARCH --resource-group $RESOURCE_GROUP --location $AI_REGION  --sku basic

#Deploy text embedding models
# az cognitiveservices account deployment create --name $AZURE_OPENAI_NAME --resource-group $RESOURCE_GROUP --deployment-name embeddings --model-name text-embedding-ada-002 --model-version 2 --sku S0
az cognitiveservices account deployment create -g $RESOURCE_GROUP -n $AZURE_OPENAI_NAME  --deployment-name embeddings  --model-name text-embedding-ada-002 --model-version "2" --model-format OpenAI --sku-capacity 1 --sku-name "Standard"

#Deploy chat completion models
#az cognitiveservices account deployment create --name $AZURE_OPENAI_NAME --resource-group $RESOURCE_GROUP --deployment-name completions --model-name gpt-4-turbo --model-version 0163 --sku S0
az cognitiveservices account deployment create -g $RESOURCE_GROUP -n $AZURE_OPENAI_NAME  --deployment-name completions  --model-name gpt-4-turbo --model-version "0125" --model-format OpenAI --sku-capacity 1 --sku-name "Standard"



echo "Environment setup complete!"
